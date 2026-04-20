from collections import Counter, defaultdict, deque
from functools import partial
from typing import BinaryIO, Callable, Iterable, List, Tuple
import gc
import gzip
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import statistics



class LogLine:

    _TEST_REGEX = ""
    _DATA_REGEX = ""

    _TEST_REGEX_COMPILED = re.compile(_TEST_REGEX)
    _DATA_REGEX_COMPILED = re.compile(_DATA_REGEX)

    subLineTypes: List[type["LogLine"]] = []

    @classmethod
    def createSubLineType(cls, LineType):
        assert(issubclass(LineType, cls))
        LineType._TEST_REGEX_COMPILED = re.compile(LineType._TEST_REGEX)
        LineType._DATA_REGEX_COMPILED = re.compile(LineType._DATA_REGEX)
        LineType.subLineTypes = []
        cls.subLineTypes.append(LineType)
        return LineType
    
    @classmethod
    def testRegex(cls, text: str) -> bool:
        return cls._TEST_REGEX_COMPILED.match(text) is not None
        
    @classmethod
    def dataRegex(cls, text: str) -> Tuple[str, ...]:
        m = cls._DATA_REGEX_COMPILED.match(text)
        if m is None:
            raise ValueError(
                f"{cls.__name__}.dataRegex match failed!"
                f" Tried to match\n\t'{text}'\n with\n\t'{cls._DATA_REGEX}'"
            )
        return m.groups()
        
    @classmethod
    def deduceLineType(cls, line: str) -> type["LogLine"] | None:
        if not cls.testRegex(line):
            return None
        matchingSubLineTypes = {LineType.deduceLineType(line) for LineType in cls.subLineTypes}
        matchingSubLineTypes.discard(None)
        if len(matchingSubLineTypes) == 0:
            return cls
        elif len(matchingSubLineTypes) == 1:
            return matchingSubLineTypes.pop()
        else:
            raise ValueError("LogLine.deduceLineType: multiple line types match: ", matchingSubLineTypes)
        
    def __init__(self, line: str) -> None:
        #self.line = line
        self.discard = False
        self.warning = False
        self.discardReason = None
        self.warningReason = None

    def discardIf(self, pred: bool, reason: str = "NoReason") -> bool:
        if pred:
            self.discard = True
            self.discardReason = reason
        return pred
    
    def warnIf(self, pred: bool, reason: str = "NoReason") -> bool:
        if pred:
            self.warning = True
            self.warningReason = reason
        return pred

    def preProcess(self, before: Iterable["LogLine"], after: Iterable["LogLine"]) -> None:
        pass

    def postProcess(self, before: Iterable["LogLine"], after: Iterable["LogLine"]) -> None:
        pass

    def endProcess(self) -> None:
        pass

    def getTotals(self) -> dict[str, int]:
        rt = {
            "total"   : int(not self.discard),
            "discard" : int(self.discard),
            "warning" : int(self.warning),
        }
        if self.discard:
            rt[f"discard({self.discardReason})"] = 1
        if self.warning:
            rt[f"warning({self.warningReason})"] = 1
        return rt

    def getDistributions(self) -> dict[str, int]:
        return {}



@LogLine.createSubLineType
class TimestampedLine(LogLine):

    _TEST_REGEX = r"^\s*\d+"
    _DATA_REGEX = r"^\s*(\d+)"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = TimestampedLine.dataRegex(line)
        self.timestamp = int(reData[0])

    def preProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().preProcess(before, after)
        
        # Fix any timestamps that appear to have used $time instead of cur_cycles
        for ll in reversed(before):
            if isinstance(ll, TimestampedLine):
                if self.timestamp >= ll.timestamp * 10 and self.timestamp % 10 == 0:
                    self.timestamp = self.timestamp // 10
                break

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Check that timestamps are monotonically increasing.
        # It is a pretty fundamental problem if this is not the case, so raise an error.
        for ll in reversed(before):
            if isinstance(ll, TimestampedLine):
                if ll.timestamp > self.timestamp:
                    raise ValueError(f"TimestampedLine.postProcess: timestamps are not monotonically increasing: {ll.timestamp} -> {self.timestamp}")
                break



@TimestampedLine.createSubLineType
class RVFILine(TimestampedLine):
    
    _TEST_REGEX = r"^\d+: RVFI Order"
    _DATA_REGEX = r"^\d+: RVFI Order: \s*(\d+), PC: (0x[0-9a-f]+), I: (0x[0-9a-f]+), PCWD: (0x[0-9a-f]+), Trap: ([01]), RD: \s*(\d+)((?:, RWD: 0x[0-9a-f]+)?)((?:, MA: 0x[0-9a-f]+)?)((?:, MWD: 0x[0-9a-f]+)?)((?:, MRM: 0b[01]+)?)((?:, MWM: 0b[01]+)?)"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = RVFILine.dataRegex(line)
        self.rvfi      = int(reData[0])
        self.pc        = int(reData[1], 0)
        self.instr     = int(reData[2], 0)
        self.pcwd      = int(reData[3], 0)
        self.trap      = bool(reData[4] == "1")
        self.rd        = int(reData[5])
        self.rwd       = int(reData[6].split(' ')[-1], 0) if reData[6] else None
        self.ma        = int(reData[7].split(' ')[-1], 0) if reData[7] else None
        self.mwd       = int(reData[8].split(' ')[-1], 0) if reData[8] else None
        self.mrm       = int(reData[9].split(' ')[-1], 0) if reData[9] else None
        self.mwm       = int(reData[10].split(' ')[-1], 0) if reData[10] else None

    def getDistributions(self):
        return {} if self.discard else {
            "pc": self.pc
        }
    
    def preProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().preProcess(before, after)

        if self.rvfi >= 2**23:
            raise ValueError(f"RVFILine.preProcess: RVFI expected to loop at 2**23 bits")

        for ll in reversed(before):
            if isinstance(ll, RVFILine):
                while self.rvfi < ll.rvfi:
                    self.rvfi += 2**23
                if ll.rvfi + 1 != self.rvfi:
                    raise ValueError(f"RVFILine.preProcess: RVFI indices are not monotonically increasing: {ll.rvfi} -> {self.rvfi}")
                break



@TimestampedLine.createSubLineType
class NonRVFILine(TimestampedLine):

    _TEST_REGEX = r"^(?!\d+: RVFI Order)"

        

@NonRVFILine.createSubLineType
class CRqCreationLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ L1D cRq creation"
    _DATA_REGEX = r"^\d+ L1D cRq creation: mshr: (\d+), addr: (0x[0-9a-f]+), vpn: (0x[0-9a-f]+), pcHash: (0x[0-9a-f]+), mshrInUse: \s*(\d+)/\s*(\d+), isPrefetch: ([01]), isRetry: ([01]), reqCs: ([ITSEM]), op: (Ld|St|Lr|Sc|Amo)"

    # How many cycles in the future to expect a cRq response
    MAX_CRQ_RESP_CYCLES = 30

    # How many cycles before a prefetch was even issued to look for a miss for this address
    MAX_LATE_PREFETCH_ISSUE_CYCLES = 20

    # To lookout for how much of capabilities we are accessing
    # class CapabilityLookout:
    #     def __init__(self, timestamp, boundsVirtBase, boundsLength):
    #         self.timestamp = timestamp
    #         self.base = boundsVirtBase
    #         self.length = boundsLength
    #         # Get the number of cache lines that the bounds cover
    #         self.nCacheLines = 1
    #         boundsLength = max(0, boundsLength - (64 - (boundsVirtBase & 0b111111)))
    #         self.nCacheLines += math.ceil(boundsLength / 64)
    #         # Create a bitmap for accesses
    #         self.lineAccessed = 0
    #
    #     # For when the capability is accessed
    #     def recordAccess(self, offset):
    #         line = ((self.base + offset) >> 6) - (self.base >> 6)
    #         self.lineAccessed |= (1 << line)
    #
    #     # Get the fraction accessed
    #     def getAccessFraction(self):
    #         return self.lineAccessed.bit_count() / self.nCacheLines
    #
    #     def reset(self, timestamp):
    #         self.timestamp = timestamp
    #         self.lineAccessed = 0
    #     def renew(self, timestamp):
    #         self.timestamp = timestamp
        
    # The lookout dict for capabilities
    MAX_CAP_USAGE_CYCLES = 1000
    CAP_USAGE_LOOKOUT = {}

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CRqCreationLine.dataRegex(line)
        self.mshr         = int(reData[0])
        self.addr         = int(reData[1], 0)
        self.lineAddr     = self.addr >> 6
        self.vpn          = int(reData[2], 0)
        self.pcHash       = int(reData[3], 0)
        self.mshrUsed     = int(reData[4])
        self.totalMshr    = int(reData[5])
        self.isPrefetch   = int(reData[6] == "1")
        self.isDemand     = not self.isPrefetch
        self.isRetry      = int(reData[7] == "1")
        self.reqCs        = str(reData[8])
        self.op           = str(reData[9])
        # Will be set in self.postProcess
        self.hit    = False
        self.miss   = False
        self.owned  = False
        self.queued = False
        self.cRqResponseLine = None
        self.cRqHitLine      = None
        # Will be set in self.postProcess if the creation is a prefetch
        self.isLatePrefetch = False
        self.prefetchLeadTime = None
        # This attribute may be set by other CRqCreationLine's in post processing
        self.isPrefetchUnderPrefetch = False
        # This attribute is set by CRqMissLine for an evicted, unused prefetch
        self.isNeverAccessed = False
        self.isNeverAccessedBecausePerms = False
        # Set by the eventual CRqHitLine
        self.disruptedCache   = False
        self.disruptionCycles = None
        # Fraction access to this capability to report in distributions
        self.accessFraction = None
        # Set by CDPFilterMissLine.postProcess when this prefetch is linked to a CDP filter-miss
        self.cdpLinked = False

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Look for the cRq response coming out of the pipeline.
        # We will then know whether we had a cache hit or miss.
        # Also look for whether this prefetch is late.
        for ll in after:
            if isinstance(ll, TimestampedLine):
                if ll.timestamp > self.timestamp + self.MAX_CRQ_RESP_CYCLES:
                    break
                # Address hit in the cache
                if isinstance(ll, CRqHitLine) and ll.mshr == self.mshr:
                    if self.discardIf(ll.addr != self.addr or ll.cRqIsPrefetch != self.isPrefetch or ll.wasMiss, "strange cRq response (hit)"): return                 
                    self.hit = True
                    self.cRqResponseLine = ll
                    ll.cRqCreationLine   = self
                    break
                # Address missed in the cache
                if isinstance(ll, CRqMissLine) and ll.mshr == self.mshr:
                    if self.discardIf(ll.addr != self.addr or ll.cRqIsPrefetch != self.isPrefetch, "strange cRq response (miss)"): return           
                    self.miss = True
                    self.cRqResponseLine = ll
                    ll.cRqCreationLine   = self
                    break
                # Cache line for the address is owned
                if isinstance(ll, CRqDependencyLine) and ll.mshr == self.mshr:
                    if self.discardIf(ll.addr != self.addr or ll.cRqIsPrefetch != self.isPrefetch, "strange cRq response (owned)"): return           
                    self.owned = True
                    self.cRqResponseLine = ll
                    ll.cRqCreationLine   = self
                    break
                # The set is completely busy, so this request is queued
                if isinstance(ll, CRqQueuedLine) and ll.mshr == self.mshr:
                    if self.discardIf(ll.addr != self.addr, "strange cRq response (queued)"): return           
                    self.queued = True
                    self.cRqResponseLine = ll
                    ll.cRqCreationLine   = self
                    break
                # This is a prefetch and another prefetch occurred for the same address.
                # Tell that prefetch that it's a duplicate.
                if isinstance(ll, CRqCreationLine) and self.isPrefetch and ll.isPrefetch and ll.lineAddr == self.lineAddr:
                    ll.isPrefetchUnderPrefetch = True
                # This is a late prefetch: there is a demand cache miss for the same address in a different MSHR.
                # This is quite unlikely: needs the core to request the address a cycle before prefetch creation (I think....)
                if isinstance(ll, CRqMissLine) and self.isPrefetch and not ll.cRqIsPrefetch and ll.newLineAddr == self.lineAddr and ll.mshr != self.mshr and not self.isLatePrefetch:
                    self.isLatePrefetch = True
                    self.prefetchLeadTime = ll.timestamp - self.timestamp

        # If we didn't find a response, discard this line
        if self.discardIf(not self.hit and not self.miss and not self.owned, "no cRq response"):
            return

        # If this is a prefetch that hit or is dependent on another cRq, then it may have been late.       
        #if self.isPrefetch and (self.hit or self.owned):
        #    for ll in reversed(before):
        #        if isinstance(ll, TimestampedLine):
        #            if ll.timestamp + self.MAX_LATE_PREFETCH_ISSUE_CYCLES < self.timestamp:
        #                break
        #            # There is an earlier miss for the same address.
        #            # If it is a demand miss, then this is a late prefetch.
        #            # If it is a prefetch miss, then this is probably a duplicate prefetch.
        #            if isinstance(ll, CRqMissLine) and ll.newLineAddr == self.lineAddr:
        #                if not ll.cRqIsPrefetch:
        #                    self.isLatePrefetch = True
        #                    self.prefetchLeadTime = ll.timestamp - self.timestamp
        #                break
        
        # If this is a prefetch that missed, then post-processing for the 
        # CRqMissLine may still deduce that this prefetch was late.  

        # If this is a prefetch for a small capability, add it to CAP_USAGE_LOOKOUT
        # if self.isPrefetch and self.boundsLength <= 512:
        #     cap = self.CAP_USAGE_LOOKOUT.get(self.boundsBase)
        #     if cap is not None:
        #         if self.timestamp > cap.timestamp + self.MAX_CAP_USAGE_CYCLES:
        #             if cap.getAccessFraction() != 0:
        #                 self.accessFraction = cap.getAccessFraction()
        #             cap.reset(self.timestamp)
        #         else:
        #             cap.renew(self.timestamp)
        #     else:
        #         self.CAP_USAGE_LOOKOUT[self.boundsBase] = self.CapabilityLookout(
        #             self.timestamp, self.boundsBase, self.boundsLength
        #         )
        #
        # # If this is a small capability, get the usage percentage
        # if self.isDemand and self.boundsLength <= 512:
        #     cap = self.CAP_USAGE_LOOKOUT.get(self.boundsBase)
        #     if cap is not None and self.timestamp < cap.timestamp + self.MAX_CAP_USAGE_CYCLES:
        #         cap.recordAccess(self.boundsOffset)

    def endProcess(self) -> None:
        super().endProcess()
        # self.accessFraction = [
        #     cap.getAccessFraction() for cap in self.CAP_USAGE_LOOKOUT.values()
        # ]
        # self.CAP_USAGE_LOOKOUT.clear()

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        return rt if self.discard else rt | {
            "Pr"    : int(self.isPrefetch),
            self.op : int(not self.isPrefetch),

            "demand"       : int(not self.isPrefetch),
            "demandHit"    : int(not self.isPrefetch and self.hit),
            "demandMiss"   : int(not self.isPrefetch and self.miss),
            "demandMissLL" : int(not self.isPrefetch and self.miss and not self.cRqResponseLine.hitInLL),
            "demandOwned"  : int(not self.isPrefetch and self.owned),
            "demandOwned"  : int(not self.isPrefetch and self.owned),
            "demandQueued" : int(not self.isPrefetch and self.queued),

            # This seems to be stuff that detected misses to Linked Data Structures
            # "demandMissToLDS"   : int(not self.isPrefetch and self.miss and self.boundsLength <= 1024),
            # "demandMissLoadToLDS" : int(not self.isPrefetch and self.miss and self.op == "Ld" and self.boundsLength <= 1024),
            # "demandMissLLToLDS"   : int(not self.isPrefetch and self.miss and not self.cRqResponseLine.hitInLL and self.boundsLength <= 1024),
            # "demandMissLLLoadToLDS"   : int(not self.isPrefetch and self.miss and not self.cRqResponseLine.hitInLL and self.op == "Ld" and self.boundsLength <= 1024),

            "prefetch"       : int(self.isPrefetch),
            "prefetchHit"    : int(self.isPrefetch and self.hit),
            "prefetchMiss"   : int(self.isPrefetch and self.miss),
            "prefetchMissLL" : int(self.isPrefetch and self.miss and not self.cRqResponseLine.hitInLL),
            "prefetchOwned"  : int(self.isPrefetch and self.owned),

            "latePrefetch" : int(self.isLatePrefetch),
            "latePrefetchCreation" : int(self.isLatePrefetch and (self.hit or self.owned)),
            "latePrefetchIssue"    : int(self.isLatePrefetch and self.miss),
            "lateUsefulPrefetch"   : int(self.isLatePrefetch and self.miss and not self.isNeverAccessed), 
            "prefUnderPref"   : int(self.isPrefetchUnderPrefetch),
            "usefulPrefetch"  : int(self.isPrefetch and self.miss and not self.isNeverAccessed),
            "uselessPrefetch" : int(self.isPrefetch and self.miss and self.isNeverAccessed),
            "uselessPrefetchBecausePerms" : int(self.isPrefetch and self.miss and self.isNeverAccessed and self.isNeverAccessedBecausePerms),
            "uselessPrefetchDisruption"   : int(self.isPrefetch and self.miss and self.isNeverAccessed and self.disruptedCache),
            "prefetchDisruption"          : int(self.isPrefetch and self.miss and self.disruptedCache),
        }
        
    def getDistributions(self) -> dict[str, int]:
        if self.discard:
            return {}
        rt = {
            "mshrRemaining" : self.totalMshr - self.mshrUsed
        }
        if not self.isPrefetch and self.cRqHitLine is not None:
            rt["demandHitCycles"] = self.cRqHitLine.timestamp - self.timestamp
        if self.isLatePrefetch:
            rt["latePrefetchWhenWasDemandMissRelativeToPrefetchCreation"] = self.prefetchLeadTime
            if self.cRqHitLine is not None: 
                rt["latePrefetchHowMuchEarlierToHit"] = self.cRqHitLine.timestamp - self.timestamp - self.prefetchLeadTime
        if self.prefetchLeadTime is not None:
            rt["prefetchLeadTime"] = self.prefetchLeadTime
        # if not self.isPrefetch:
        #     rt["demandCapSize"] = self.boundsLength
        # if self.isPrefetch:
        #     rt["prefetchCapSize"] = self.boundsLength
        # if not self.isPrefetch and self.cRqHitLine is not None and self.cRqHitLine.nCap > 0:
        #     rt["demandHasPtrsCapSize"] = self.boundsLength
        # if self.accessFraction:
        #     rt["smallCapAccessFraction"] = self.accessFraction
        return rt

@NonRVFILine.createSubLineType
class LLCRqCreationLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ LL cRq creation"
    _DATA_REGEX = r"^\d+ LL cRq creation: mshr: \s*(\d+), addr: (0x[0-9a-f]+), vpn: (0x[0-9a-f]+), mshrInUse: \s*(\d+)/\s*(\d+), isPrefetch: ([01]), wasQueued: ([01]), reqCs: ([ITSEM])"
    
    # Max cycles to search for a cache hit
    MAX_HIT_CYCLES = 500

    # Max previous cycles to search for a demand miss
    MAX_LATE_PREFETCH_ISSUE_CYCLES = 20

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = LLCRqCreationLine.dataRegex(line)
        self.mshr         = int(reData[0])
        self.addr         = int(reData[1], 0)
        self.lineAddr     = self.addr >> 6
        self.vpn          = int(reData[2], 0)
        self.mshrUsed     = int(reData[3])
        self.totalMshr    = int(reData[4])
        self.isPrefetch   = bool(reData[5] == "1")
        self.isDemand     = not self.isPrefetch
        self.isRetry      = bool(reData[6] == "1")
        self.reqCs        = str(reData[7])
        # Will be set in self.postProcess
        self.cRqResponseLine = None
        self.owned = False
        self.miss  = False
        self.hit   = False
        self.cRqHitLine  = None
        # Will be set in self.postProcess if the creation is a prefetch
        self.isLatePrefetch = False
        self.prefetchLeadTime = None
        # This attribute is set by LLCRqMissLine for an evicted, unused prefetch
        self.isNeverAccessed = False

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Look for a LLCRqMissLine and then the eventual LLCRqHitLine
        for ll in after:
            if isinstance(ll, TimestampedLine) and ll.timestamp > self.timestamp + self.MAX_HIT_CYCLES:
                self.discardIf(True, "no LL cRq hit")
                break
            if isinstance(ll, LLCRqDependencyLine) and ll.mshr == self.mshr:
                if self.discardIf(ll.addr != self.addr, "strange LL cRq response (owned)"): return
                self.cRqResponseLine = ll
                ll.cRqCreationLine = self
                # Prefetches are dropped at this point
                self.owned = True
                if self.isPrefetch:
                    break
            if isinstance(ll, LLCRqMissLine) and ll.mshr == self.mshr:
                if self.discardIf(ll.addr != self.addr, "strange LL cRq response (miss)"): return
                self.cRqResponseLine = ll
                ll.cRqCreationLine = self
                self.miss = True
            if isinstance(ll, LLCRqHitLine) and ll.mshr == self.mshr:
                if self.discardIf(ll.addr != self.addr, "strange LL cRq response (hit)"): return
                self.cRqResponseLine = ll
                self.cRqHitLine = ll
                ll.cRqCreationLine = self
                if not self.miss and not self.owned:
                    self.hit = True
                break
            # Looking for a hit on this prefetch
            if isinstance(ll, LLCRqMissLine) and self.isPrefetch and not ll.cRqIsPrefetch and ll.newLineAddr == self.lineAddr and ll.mshr != self.mshr and not self.isLatePrefetch:
                self.isLatePrefetch = True
                self.prefetchLeadTime = ll.timestamp - self.timestamp
            if isinstance(ll, LLCRqDependencyLine) and self.isPrefetch and not ll.cRqIsPrefetch and ll.lineAddr == self.lineAddr and ll.mshr != self.mshr and not self.isLatePrefetch:
                self.isLatePrefetch = True
                self.prefetchLeadTime = ll.timestamp - self.timestamp

        # Check whether this is a late prefetch
        #if self.isPrefetch and (self.hit or self.owned):
        #    for ll in reversed(before):
        #        if isinstance(ll, TimestampedLine):
        #            if ll.timestamp + self.MAX_LATE_PREFETCH_ISSUE_CYCLES < self.timestamp:
        #                break
        #            if isinstance(ll, CRqMissLine) and ll.newLineAddr == self.lineAddr:
        #                if not ll.cRqIsPrefetch:
        #                    self.isLatePrefetch = True
        #                    self.prefetchLeadTime = ll.timestamp - self.timestamp
        #                break
        
        # If we didn't find a response, discard this line
        if self.discardIf(not self.hit and not self.miss and not self.owned, "no LL cRq response"):
            return

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        return rt if self.discard else rt | {
            "demand"      : int(not self.isPrefetch),
            "demandHit"   : int(not self.isPrefetch and self.hit),
            "demandMiss"  : int(not self.isPrefetch and self.miss),
            "demandOwned" : int(not self.isPrefetch and self.owned),

            "prefetch"      : int(self.isPrefetch),
            "prefetchHit"   : int(self.isPrefetch and self.hit),
            "prefetchMiss"  : int(self.isPrefetch and self.miss),
            "prefetchOwned" : int(self.isPrefetch and self.owned),
            "prefetchOwnedAddr" : int(self.isPrefetch and self.owned and self.cRqResponseLine.addrSucc),
            "prefetchOwnedDemandAddr" : int(self.isPrefetch and self.owned and self.cRqResponseLine.addrSucc and not self.cRqResponseLine.ownerIsPrefetch),
            "prefetchOwnedRep"  : int(self.isPrefetch and self.owned and not self.cRqResponseLine.addrSucc),

            "usefulPrefetch"  : int(self.isPrefetch and self.miss and not self.isNeverAccessed),
            "uselessPrefetch" : int(self.isPrefetch and self.miss and self.isNeverAccessed),
        }
    
    def getDistributions(self) -> dict[str, int]:
        if self.discard:
            return {}
        rt = {
            "mshrRemaining" : self.totalMshr - self.mshrUsed
        }
        if not self.isPrefetch and self.cRqHitLine is not None:
            rt["demandHitCycles"] = self.cRqHitLine.timestamp - self.timestamp
        if self.isLatePrefetch:
            rt["latePrefetchWhenWasDemandMissRelativeToPrefetchCreation"] = self.prefetchLeadTime
            if self.cRqHitLine is not None: 
                rt["latePrefetchHowMuchEarlierToHit"] = self.cRqHitLine.timestamp - self.timestamp - self.prefetchLeadTime
        if self.prefetchLeadTime is not None:
            rt["prefetchLeadTime"] = self.prefetchLeadTime
        return rt

@NonRVFILine.createSubLineType
class CRqHitLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ L1D cRq hit"
    _DATA_REGEX = r"^\d+ L1D cRq hit: mshr: \s*(\d+), addr: (0x[0-9a-f]+), cRq is prefetch: ([01]), wasMiss: ([01]), pipeCs: ([ITSEM]), reqCs: ([ITSEM]), saveCs: ([ITSEM]), op: (Ld|St|Lr|Sc|Amo), data: (.*)"

    # Lookout for the eviction of this cache line
    # Means we can know prefetch accuracy and the lifetime of cache lines
    EVICTION_LOOKOUTS = defaultdict(set)

    # So that we can calculate prefetch lead time
    PREFETCH_LOOKOUTS = dict()

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CRqHitLine.dataRegex(line)
        self.mshr          = int(reData[0])
        self.addr          = int(reData[1], 0)
        self.lineAddr      = self.addr >> 6
        self.cRqIsPrefetch = bool(reData[2] == "1")
        self.wasMiss       = bool(reData[3] == "1")
        self.pipeCs        = str(reData[4])
        self.reqCs         = str(reData[5])
        self.saveCs        = str(reData[6])
        self.op            = str(reData[7])
        self.lineData      = str(reData[8])
        # Will be set by a prior CRqCreationLine if this is an immediate hit,
        # or a CRqMissLine/CRqDependencyLine otherwise.
        self.cRqCreationLine = None
        # Will be set by a prior CRqMissLine if this hit was once a miss
        self.cRqMissLine = None
        # Set in self.postProcess
        self.evictionLine   = None
        # The data being loaded (if a load)
        self.hitDataLine = None

    def postProcess(self, before, after):
        super().postProcess(before, after)
        if self.discard: return

        # Produce a warning if there was no creation of this cRq
        self.warnIf(self.cRqCreationLine is None, "no creation of cRq")
        if self.cRqCreationLine is not None:
            self.cRqCreationLine.cRqHitLine = self

        # If this is the refill of a miss, then look for the eventual eviction.
        if self.wasMiss:
            self.warnIf(self.cRqMissLine is None, "no cRq eviction for miss")
            self.EVICTION_LOOKOUTS[self.lineAddr].add(self)

        # Find the data being loaded. Discard if not found.
        if self.op == "Ld" and not self.cRqIsPrefetch:
            for ll in after:
                if isinstance(ll, CRqHitDataLine):
                    self.hitDataLine = ll
                    break
            if self.discardIf(self.hitDataLine == None, "no data for load hit (could be loading tags)"): return
        
        # If this is a prefetch, add it as a lookout
        if self.cRqIsPrefetch and self.cRqCreationLine is not None and self.wasMiss:
            self.PREFETCH_LOOKOUTS[self.lineAddr] = self.cRqCreationLine
        
        # If this was a demand hit, check whether there was a previous prefetch for this line
        # If so, set its lead time
        if not self.cRqIsPrefetch and self.lineAddr in self.PREFETCH_LOOKOUTS:
            if not self.warnIf(self.wasMiss, "CRqHitLine.PREFETCH_LOOKOUTS hit on a cRq miss"):
                ll = self.PREFETCH_LOOKOUTS[self.lineAddr]
                if ll.prefetchLeadTime is None:
                    ll.prefetchLeadTime = self.timestamp - ll.timestamp
                self.PREFETCH_LOOKOUTS.pop(self.lineAddr)

        # Check whether this line is being demanded back into the cache after it was removed by a prefetch
        if not self.cRqIsPrefetch:
            ll = CRqMissLine.CACHE_DISRUPTION_EVICTEE_LOOKOUT.get(self.lineAddr)
            if ll is not None:
                assert(ll.oldLineAddr == self.lineAddr)
                ll.cRqCreationLine.disruptedCache = True
                CRqMissLine.CACHE_DISRUPTION_EVICTEE_LOOKOUT.pop(ll.oldLineAddr)
                if ll.newLineAddr in CRqMissLine.CACHE_DISRUPTION_PREFETCH_LOOKOUT:
                    CRqMissLine.CACHE_DISRUPTION_PREFETCH_LOOKOUT.pop(ll.newLineAddr)
                else:
                    self.warnIf(True, "CACHE_DISRUPTION_PREFETCH_LOOKOUT entry missing")

    def getTotals(self):
        rt = super().getTotals()
        return rt if self.discard else rt | {
            "demandAccesses": (self.cRqCreationLine is not None and not self.cRqIsPrefetch and self.op == "Ld" and self.addr % 16 == 0)
            # AlexNote: previously in CHERI ver we could see if the demand access was against a capability, of course this isn't possible now
            #"demandAccessCap": (self.cRqCreationLine is not None and not self.cRqIsPrefetch and self.op == "Ld" and self.addr % 16 == 0 and self.hitDataLine.tag and self.cRqCreationLine.boundsLength >= 16)
        }

    def getDistributions(self):
        if self.discard:
            return {}
        rt = {}
        if not self.cRqIsPrefetch:
            rt["demandAddr"] = self.addr
        # if self.cRqCreationLine is not None and not self.cRqCreationLine.isPrefetch:
        #     rt["demandNCap"] = self.nCap
        # if self.wasMiss:
        #     rt["missNCap"] = self.nCap
        # if self.cRqCreationLine is not None and not self.cRqIsPrefetch and self.op == "Ld" and self.addr % 16 == 0 and self.hitDataLine.tag:
        #     rt["demandCapSizeForCapLoad"] = self.cRqCreationLine.boundsLength
        # if self.cRqCreationLine is not None and not self.cRqIsPrefetch and self.op == "Ld" and self.addr % 16 == 0 and self.hitDataLine.tag and self.cRqCreationLine.boundsLength >= 16:
        #     rt["demandCapSizeForSensibleCapLoad"] = self.cRqCreationLine.boundsLength
        if self.wasMiss and self.evictionLine is not None:
            rt["evictionCycles"] = self.evictionLine.timestamp - self.timestamp
        # if self.wasMiss and self.evictionLine is not None and self.cRqCreationLine is not None and self.cRqCreationLine.boundsLength <= 512:
        #     rt["smallCapEvictionCycles"] = self.evictionLine.timestamp - self.timestamp
        return rt

@NonRVFILine.createSubLineType
class CRqHitDataLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ L1Bank hit data:"
    # _DATA_REGEX = r"^\d+ L1Bank hit data: TaggedData { tag: (True|False), data: <V 'h([0-9a-f]+) 'h([0-9a-f]+)  > }"
    _DATA_REGEX = r"^\d+ L1Bank hit data: 'h([0-9a-f]+)"

    def __init__(self, line: str):
        super().__init__(line)
        reData = CRqHitDataLine.dataRegex(line)
        self.data0 = int(reData[0], 16)

@NonRVFILine.createSubLineType
class LLCRqHitLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ LL cRq hit"
    _DATA_REGEX = r"^\d+ LL cRq hit mshr: \s*(\d+), addr: (0x[0-9a-f]+), cRq is prefetch: ([01]), wasMiss: ([01])"

    # Lookout for the eviction of this cache line
    # Means we can know prefetch accuracy and the lifetime of cache lines
    EVICTION_LOOKOUTS = defaultdict(set)

    # So that we can calculate prefetch lead time
    PREFETCH_LOOKOUTS = dict()

    def __init__(self, line: str):
        super().__init__(line)
        reData = LLCRqHitLine.dataRegex(line)
        self.mshr          = int(reData[0])
        self.addr          = int(reData[1], 0)
        self.lineAddr      = self.addr >> 6
        self.cRqIsPrefetch = bool(reData[2] == "1")
        self.wasMiss       = bool(reData[3] == "1")
        # Will be set by a prior LLCRqCreationLine
        self.cRqCreationLine = None
        # Will be set by a LLCRqMissLine
        self.evictionLine    = None

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Warn that there is no creation
        self.warnIf(self.cRqCreationLine is None, "no creation of LL cRq")

        # If this is the refill of a miss, then look for the eventual eviction.
        if self.wasMiss:
            self.EVICTION_LOOKOUTS[self.lineAddr].add(self)

        # If this is a prefetch, add it as a lookout
        if self.cRqIsPrefetch and self.cRqCreationLine is not None and self.wasMiss:
            self.PREFETCH_LOOKOUTS[self.lineAddr] = self.cRqCreationLine

        # If this was a demand hit, check whether there was a previous prefetch for this line
        # If so, set its lead time
        if not self.cRqIsPrefetch and self.lineAddr in self.PREFETCH_LOOKOUTS:
            if not self.warnIf(self.wasMiss, "LLCRqHitLine.PREFETCH_LOOKOUTS hit on a cRq miss"):
                ll = self.PREFETCH_LOOKOUTS[self.lineAddr]
                if ll.prefetchLeadTime is None:
                    ll.prefetchLeadTime = self.timestamp - ll.timestamp
                self.PREFETCH_LOOKOUTS.pop(self.lineAddr)
    
    def getDistributions(self):
        if self.discard:
            return {}
        rt = {}
        if self.wasMiss and self.evictionLine is not None:
            rt["evictionCycles"] = self.evictionLine.timestamp - self.timestamp
        # if self.wasMiss and self.evictionLine is not None and self.cRqCreationLine is not None and self.cRqCreationLine.boundsLength <= 512:
        #     rt["smallCapEvictionCycles"] = self.evictionLine.timestamp - self.timestamp
        return rt

@NonRVFILine.createSubLineType
class CRqMissLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ L1D cRq miss"
    _DATA_REGEX = r"^\d+ L1D cRq miss \(([\w ]+)\): mshr: (\d+), addr: (0x[0-9a-f]+), old line addr: (0x[0-9a-f]+), wasPrefetch: ([01]), cRq is prefetch: ([01]), ramCs: ([ITSEM]), reqCs: ([ITSEM]), op: (Ld|St|Lr|Sc|Amo)"

    # How many cycles after a cRq miss to expect the corresponding cache refill
    MAX_PRS_CYCLES = 300

    # So that we can detect cache disruption
    CACHE_DISRUPTION_PREFETCH_LOOKOUT = dict()
    CACHE_DISRUPTION_EVICTEE_LOOKOUT = dict()

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CRqMissLine.dataRegex(line)
        self.repType       = str(reData[0])
        self.mshr          = int(reData[1])
        self.addr          = int(reData[2], 0)
        self.newLineAddr   = self.addr >> 6
        self.oldLineAddr   = int(reData[3], 0)
        self.wasPrefetch   = bool(reData[4] == "1")
        self.cRqIsPrefetch = bool(reData[5] == "1")
        self.ramCs         = str(reData[6])
        self.reqCs         = str(reData[7])
        self.op            = str(reData[8])
        self.permsOnly     = self.oldLineAddr == self.newLineAddr and self.ramCs != "I"
        # Set in self.postProcess
        self.hitInLL       = None
        self.cRqHitLine    = None
        # Will be set by a prior CRqCreationLine
        self.cRqCreationLine = None

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Produce a warning if there was no creation of this cRq
        self.warnIf(self.cRqCreationLine is None, f"no creation of cRq ({self.repType})")

        # Discard self and creation line
        def discardWithCreationIf(cond: bool, reason: str):
            if self.discardIf(cond, reason) and self.cRqCreationLine is not None:
                self.cRqCreationLine.discardIf(True, reason)
            return cond

        # Look for the cache refill. When a pRs is received for the refill data, the cRqHit rule fires.
        # Also look for the LL hit line (which will tell us if the LL hit or missed).
        # If this cRq is for a prefetch, also see if this prefetch was late, or whether there is a prefetch-under-prefetch.
        foundLLHit  = False
        foundRefill = False
        for ll in after:
            if(isinstance(ll, TimestampedLine)):
                if ll.timestamp > self.timestamp + self.MAX_PRS_CYCLES:
                    break
                # The LL hit line. Assume only one core.
                if isinstance(ll, LLCRqHitLine) and ll.addr == self.addr and not ll.cRqIsPrefetch:
                    if discardWithCreationIf(foundLLHit, "multiple LL hits"): return
                    foundLLHit   = True
                    self.hitInLL = not ll.wasMiss
                # The refill for the miss
                if isinstance(ll, CRqHitLine) and ll.mshr == self.mshr:
                    if discardWithCreationIf(ll.addr != self.addr or ll.cRqIsPrefetch != self.cRqIsPrefetch or not ll.wasMiss, "strange refill"): return
                    foundRefill = True
                    ll.cRqMissLine     = self
                    ll.cRqCreationLine = self.cRqCreationLine
                    self.cRqHitLine    = ll
                    break
                # This is a prefetch and another prefetch occurred for the same address during refill.
                # Tell that prefetch that it's a duplicate.
                if isinstance(ll, CRqCreationLine) and self.cRqIsPrefetch and ll.isPrefetch and ll.lineAddr == self.newLineAddr:
                    ll.isPrefetchUnderPrefetch = True
                # Look for a demand miss for the address currently being refilled.
                # If the current refill is due to a prefetch, then consider the prefetch as late.
                if isinstance(ll, CRqDependencyLine) and self.cRqIsPrefetch and not ll.cRqIsPrefetch and ll.lineAddr == self.newLineAddr:
                    # It's possible that, if some log was skipped, the cRq is a prefetch but the creation was missed.
                    if self.cRqCreationLine is not None and not self.cRqCreationLine.isLatePrefetch:
                        self.cRqCreationLine.isLatePrefetch = True
                        self.cRqCreationLine.prefetchLeadTime = ll.timestamp - self.cRqCreationLine.timestamp

        # If we didn't find an L1 or LL hit line, discard this log line and the cRq creator.
        if discardWithCreationIf(not foundRefill, "no pRs"): return
        if discardWithCreationIf(not foundLLHit, "pRs with no LL hit"): return 

        # Check in the eviction lookouts of CRqHitLine
        if self.oldLineAddr in CRqHitLine.EVICTION_LOOKOUTS:
            for ll in CRqHitLine.EVICTION_LOOKOUTS[self.oldLineAddr]:
                # Maybe tell the creation log line that it was never accessed
                # Note that wasPrefetch is unset when the line is accessed
                if self.wasPrefetch and ll.cRqCreationLine is not None and not ll.cRqCreationLine.isNeverAccessed:
                    ll.cRqCreationLine.isNeverAccessed = True
                    ll.cRqCreationLine.isNeverAccessedBecausePerms |= self.permsOnly
                if not self.permsOnly:
                    ll.evictionLine = self
            # The miss could actually be a permissions upgrade: don't count this as eviction
            if not self.permsOnly:
                CRqHitLine.EVICTION_LOOKOUTS[self.oldLineAddr].clear()

        # Remove any stale prefetch lookout
        if self.oldLineAddr in CRqHitLine.PREFETCH_LOOKOUTS:
            CRqHitLine.PREFETCH_LOOKOUTS.pop(self.oldLineAddr)

        # Check if we're evicting a prefetch, because we should remove it from CACHE_DISRUPTION_PREFETCH_LOOKOUT
        if self.oldLineAddr in self.CACHE_DISRUPTION_PREFETCH_LOOKOUT:
            ll = self.CACHE_DISRUPTION_PREFETCH_LOOKOUT[self.oldLineAddr]
            assert(ll.newLineAddr == self.oldLineAddr)
            self.CACHE_DISRUPTION_PREFETCH_LOOKOUT.pop(ll.newLineAddr)
            # It's possible that ll.oldLineAddr made it's way back into the cache via a prefetch, and then got evicted again
            # In this case, CACHE_DISRUPTION_EVICTEE_LOOKOUT[ll.oldLineAddr] will point to a different prefetch
            if self.CACHE_DISRUPTION_EVICTEE_LOOKOUT.get(ll.oldLineAddr, None) == ll:
                self.CACHE_DISRUPTION_EVICTEE_LOOKOUT.pop(ll.oldLineAddr)

        # If this is a prefetch, create lookouts to see whether we evicted a useful cache line
        if self.cRqIsPrefetch and self.cRqCreationLine is not None and not self.permsOnly:
            self.CACHE_DISRUPTION_PREFETCH_LOOKOUT[self.newLineAddr] = self
            self.CACHE_DISRUPTION_EVICTEE_LOOKOUT[self.oldLineAddr] = self

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard:
            return rt
        if self.oldLineAddr == self.newLineAddr:
            rt[f"{self.ramCs} --({'Pr' if self.cRqIsPrefetch else self.op})--> {self.reqCs}"] = 1
        else:
            rt[f"{self.ramCs} --/{'Pr' if self.cRqIsPrefetch else self.op}/--> {self.reqCs}"] = 1
        return rt
    
    def getDistributions(self) -> dict[str, int]:
        if self.discard:
            return {}
        rt = {"refillCycles" : self.cRqHitLine.timestamp - self.timestamp}
        return rt

@NonRVFILine.createSubLineType
class LLCRqMissLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ LL cRq miss"
    _DATA_REGEX = r"^\d+ LL cRq miss \(([\w ]+)\): mshr: \s*(\d+), addr: (0x[0-9a-f]+), old line addr: (0x[0-9a-f]+), wasPrefetch: ([01]), cRq is prefetch: ([01]), ramCs: ([ITSEM]), reqCs: ([ITSEM])"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = LLCRqMissLine.dataRegex(line)
        self.repType       = str(reData[0])
        self.mshr          = int(reData[1])
        self.addr          = int(reData[2], 0)
        self.newLineAddr   = self.addr >> 6
        self.oldLineAddr   = int(reData[3], 0)
        self.wasPrefetch   = bool(reData[4] == "1")
        self.cRqIsPrefetch = bool(reData[5] == "1")
        self.ramCs         = str(reData[6])
        self.reqCs         = str(reData[7])
        # Will be set by a prior CRqCreationLine
        self.cRqCreationLine = None

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Warn that there is no creation
        self.warnIf(self.cRqCreationLine is None, "No creation of LL cRq")

        # Check in the eviction lookouts of CRqHitLine
        if self.oldLineAddr in LLCRqHitLine.EVICTION_LOOKOUTS:
            for ll in LLCRqHitLine.EVICTION_LOOKOUTS[self.oldLineAddr]:
                # Maybe tell the creation log line that it was never accessed
                # Note that wasPrefetch is unset when the line is accessed
                if self.wasPrefetch and ll.cRqCreationLine is not None and not ll.cRqCreationLine.isNeverAccessed:
                    ll.cRqCreationLine.isNeverAccessed = True
                ll.evictionLine = self
            LLCRqHitLine.EVICTION_LOOKOUTS[self.oldLineAddr].clear()

        # Remove any stale prefetch lookout
        if self.oldLineAddr in LLCRqHitLine.PREFETCH_LOOKOUTS:
            LLCRqHitLine.PREFETCH_LOOKOUTS.pop(self.oldLineAddr)

@NonRVFILine.createSubLineType
class CRqDependencyLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ L1D cRq dependency"
    _DATA_REGEX = r"^\d+ L1D cRq dependency: mshr: \s*(\d+), depMshr: \s*(\d+), addr: (0x[0-9a-f]+), cRq is prefetch: ([01]), reqCs: ([ITSEM]), op: (Ld|St|Lr|Sc|Amo)"

    # How many cycles after a cRq miss to expect the corresponding cache refill
    MAX_PRS_CYCLES = 300

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CRqDependencyLine.dataRegex(line)
        self.mshr          = int(reData[0])
        self.depMshr       = int(reData[1])
        self.addr          = int(reData[2], 0)
        self.lineAddr      = self.addr >> 6
        self.cRqIsPrefetch = bool(reData[3] == "1")
        self.reqCs         = str(reData[4])
        self.op            = str(reData[5])
        # Set in self.postProcess
        self.resolveHit     = False
        self.resolveMiss    = False
        self.cRqResolveLine = None
        # Will be set by a prior CRqCreationLine
        self.cRqCreationLine = None

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Produce a warning if there was no creation of this cRq
        self.warnIf(self.cRqCreationLine is None, "no creation of cRq")

        # Discard self and creation line
        def discardWithCreationIf(cond: bool, reason: str):
            if self.discardIf(cond, reason) and self.cRqCreationLine is not None:
                self.cRqCreationLine.discardIf(True, reason)
            return cond

        # If this is not a prefetch, look for the dependency to resolve. 
        # Will probably be cache hit, but might be a miss without replacement if c-state is not enough to hit.
        # If this is a prefetch, then it will be dropped here.
        if not self.cRqIsPrefetch:
            for ll in after:
                if(isinstance(ll, TimestampedLine)):
                    if ll.timestamp > self.timestamp + self.MAX_PRS_CYCLES:
                        break
                    # Dependency resolved and hit
                    if isinstance(ll, CRqHitLine) and ll.mshr == self.mshr:
                        if discardWithCreationIf(ll.addr != self.addr or ll.cRqIsPrefetch != self.cRqIsPrefetch, "strange resolution"): return
                        self.resolveHit     = True
                        self.cRqResolveLine = ll
                        ll.cRqCreationLine  = self.cRqCreationLine
                        break
                    # Dependency resolved but permissions missed
                    if isinstance(ll, CRqMissLine) and ll.mshr == self.mshr:
                        if discardWithCreationIf(ll.oldLineAddr != self.lineAddr or ll.addr != self.addr or ll.cRqIsPrefetch != self.cRqIsPrefetch, "strange resolution"): return
                        self.resolveMiss    = True
                        self.cRqResolveLine = ll
                        ll.cRqCreationLine  = self.cRqCreationLine
                        break 
                    # This is a prefetch and another prefetch occurred for the same address during resolution.
                    # Tell that prefetch that it's a duplicate.
                    if isinstance(ll, CRqCreationLine) and self.cRqIsPrefetch and ll.isPrefetch and ll.lineAddr == self.lineAddr:
                        ll.isPrefetchUnderPrefetch = True
            
            # If we didn't find a resolution, discard this log line and the cRq creator.
            if discardWithCreationIf(not self.resolveHit and not self.resolveMiss, "no resolution"): return
        
    def getTotals(self):
        rt = super().getTotals()
        return rt if self.discard else rt | {
            "demand"      : int(not self.cRqIsPrefetch),
            "resolveHit"  : int(self.resolveHit),
            "resolveMiss" : int(self.resolveMiss),
        }
    
    def getDistributions(self):
        if self.discard:
            return {}
        rt = {}
        if not self.cRqIsPrefetch:
            rt["resolveCycles"] = self.cRqResolveLine.timestamp - self.timestamp
        return rt

@NonRVFILine.createSubLineType
class LLCRqDependencyLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ LL cRq dependency"
    _DATA_REGEX = r"^\d+ LL cRq dependency \((rep|addr) succ\): mshr: \s*(\d+), depMshr: \s*(\d+), addr: (0x[0-9a-f]+), cRq is prefetch: ([01]), other is prefetch: ([01]), reqCs: ([ITSEM])"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = LLCRqDependencyLine.dataRegex(line)
        self.addrSucc        = bool(reData[0] == "addr")
        self.mshr            = int(reData[1])
        self.depMshr         = int(reData[2])
        self.addr            = int(reData[3], 0)
        self.lineAddr        = self.addr >> 6
        self.cRqIsPrefetch   = bool(reData[4] == "1")
        self.ownerIsPrefetch = bool(reData[5] == "1")
        self.reqCs           = str(reData[6])
        # Will be set by a prior LLCRqCreationLine
        self.cRqCreationLine = None

@NonRVFILine.createSubLineType
class CRqQueuedLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ L1D cRq queued"
    _DATA_REGEX = r"^\d+ L1D cRq queued: mshr: \s*(\d+), addr: (0x[0-9a-f]+), succTo: tagged (?:Valid|Invalid) (?:'h([a-zA-Z0-9]+))?, reqCs: ([ITSEM]), op: (Ld|St|Lr|Sc|Amo)"

    def __init__(self, line: str):
        super().__init__(line)
        reData = CRqQueuedLine.dataRegex(line)
        self.mshr          = int(reData[0])
        self.addr          = int(reData[1], 0)
        self.lineAddr      = self.addr >> 6
        self.succTo        = None if reData[2] is None else int(reData[2], 16)
        self.reqCs         = str(reData[3])
        self.op            = str(reData[4])
        # Will be set by a prior CRqCreationLine 
        self.cRqCreationLine = None
        # Will be set in postProcess
        self.cRqRetryLine = None

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Produce a warning if there was no creation of this cRq
        self.warnIf(self.cRqCreationLine is None, "no creation of cRq queuing")

@NonRVFILine.createSubLineType
class PRqLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ L1D pRq"
    _DATA_REGEX = r"^\d+ L1D pRq: line addr: (0x[0-9a-f]+), wasPrefetch: ([01]), overtakeCRq: ([01]), ramCs: ([ITSEM]), reqCs: ([ITSEM])"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = PRqLine.dataRegex(line)
        self.lineAddr    = int(reData[0], 0)
        self.wasPrefetch = bool(reData[1] == "1")
        self.overtakeCRq = bool(reData[2] == "1")
        self.ramCs       = str(reData[3])
        self.reqCs       = str(reData[4])

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Check in the eviction lookouts of CRqHitLine
        if self.reqCs == "I" and self.lineAddr in CRqHitLine.EVICTION_LOOKOUTS:
            for ll in CRqHitLine.EVICTION_LOOKOUTS[self.lineAddr]:
                if self.wasPrefetch and ll.cRqCreationLine is not None:
                    ll.cRqCreationLine.isNeverAccessed = True
                ll.evictionLine = self
            CRqHitLine.EVICTION_LOOKOUTS[self.lineAddr].clear()

        # Remove any stale prefetch lookout
        if self.reqCs == "I" and self.lineAddr in CRqHitLine.PREFETCH_LOOKOUTS:
            CRqHitLine.PREFETCH_LOOKOUTS.pop(self.lineAddr)

# ============================================================================
# CDP Prefetcher log line parsers
# All correspond to $display lines in mkCDPStatefulRelative (CDP.bsv)
# ============================================================================

@NonRVFILine.createSubLineType
class CDPCandidateLine(NonRVFILine):
    """Candidate pointer identified in incoming cache line (deqCacheLines rule)."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel candidate vaddr"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel candidate vaddr relOffset:\s*([+-]?\d+) pcHash: ([0-9a-f]+) candVaddr: ([0-9a-f]+) crossPage: ([01])"

    # candVaddr -> (timestamp_or_None, crossPage). timestamp is consumed on first
    # matching training hit (for the lag metric) and set to None; crossPage persists
    # for subsequent training hits on the same vaddr.
    VADDR_LOOKOUT: dict[int, tuple[int | None, bool]] = {}

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPCandidateLine.dataRegex(line)
        self.relOffset  = int(reData[1])
        self.pcHash     = int(reData[2], 16)
        self.candVaddr  = int(reData[3], 16)
        self.crossPage  = reData[4] == "1"

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return
        # Refresh timestamp even if this vaddr was seen before — gives lag-since-most-recent.
        CDPCandidateLine.VADDR_LOOKOUT[self.candVaddr] = (self.timestamp, self.crossPage)

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {
            "candidatesFound": 1,
            "crossPageCandidates": int(self.crossPage),
        }

    def getDistributions(self) -> dict[str, int]:
        if self.discard: return {}
        return {"candidateRelOffset": self.relOffset}


@NonRVFILine.createSubLineType
class CDPTrainingHitLine(NonRVFILine):
    """Training table hit: miss vaddr was previously seen as a pointer."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel Training hit"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel Training hit: missVaddr ([0-9a-f]+) seen before by pcHash ([0-9a-f]+) at relOffset\s*([+-]?\d+)"

    # Max cycles to search forward for a matching decision/no-high-conf with the same pcHash
    MAX_TT_TO_DECISION_CYCLES = 100

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPTrainingHitLine.dataRegex(line)
        self.missVaddr  = int(reData[1], 16)
        self.pcHash     = int(reData[2], 16)
        self.relOffset  = int(reData[3])
        # Set in self.postProcess
        self.resolution: str | None = None     # "decision" | "noHighConf" | None
        self.candidateLag: int | None = None
        self.originCrossPage: bool | None = None

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Part 4: look up VADDR_LOOKOUT. Record lag only on first consumption (by
        # clearing the ts); keep crossPage so downstream TT hits on the same vaddr
        # can still be categorized as same-page vs cross-page.
        entry = CDPCandidateLine.VADDR_LOOKOUT.get(self.missVaddr)
        if entry is not None:
            ts, crossPage = entry
            self.originCrossPage = crossPage
            if ts is not None:
                self.candidateLag = self.timestamp - ts
                CDPCandidateLine.VADDR_LOOKOUT[self.missVaddr] = (None, crossPage)

        # Part 3: claim every matching-pcHash decision within the window so originCrossPage
        # propagates to the full fan-out of decisions produced by this TT hit (inBounds +
        # each neighbour, one per PCT high-conf offset). Stop early only on no-high-conf,
        # which rules out any decisions from this lookup.
        for ll in after:
            if isinstance(ll, TimestampedLine) and ll.timestamp > self.timestamp + self.MAX_TT_TO_DECISION_CYCLES:
                break
            if isinstance(ll, CDPNoHighConfLine) and ll.pcHash == self.pcHash:
                if self.resolution is None:
                    self.resolution = "noHighConf"
                break
            if isinstance(ll, CDPPrefetchDecisionLine) and ll.pcHash == self.pcHash and ll.trainingHitLine is None:
                self.resolution = "decision"
                ll.trainingHitLine = self
                ll.originCrossPage = self.originCrossPage

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {
            "ttTrainingHits":             1,
            "ttHitResolvedDecision":      int(self.resolution == "decision"),
            "ttHitResolvedNoHighConf":    int(self.resolution == "noHighConf"),
            "ttHitUnresolved":            int(self.resolution is None),
            "ttHitWithPriorCandidate":    int(self.candidateLag is not None),
            "ttHitWithCrossPageOrigin":   int(self.originCrossPage is True),
            "ttHitWithSamePageOrigin":    int(self.originCrossPage is False),
        }

    def getDistributions(self) -> dict[str, int]:
        if self.discard: return {}
        rt: dict[str, int] = {"trainingHitRelOffset": self.relOffset}
        if self.candidateLag is not None:
            rt["candidateToTrainingLag"] = self.candidateLag
        return rt


@NonRVFILine.createSubLineType
class CDPTtWriteLine(NonRVFILine):
    """Training table write: new entry or overwrite of existing entry."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel (Wrote|Overwrote) to training table"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel (Wrote|Overwrote) to training table, idx:\s*\d+ candVaddr: ([0-9a-f]+)(?:.*relOffset:\s*([+-]?\d+))"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPTtWriteLine.dataRegex(line)
        self.isOverwrite = reData[1] == "Overwrote"
        self.candVaddr   = int(reData[2], 16)
        self.relOffset   = int(reData[3]) if reData[3] is not None else 0

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {
            "ttWrites":     1,
            "ttOverwrites": int(self.isOverwrite),
            "ttNewEntries": int(not self.isOverwrite),
        }

    def getDistributions(self) -> dict[str, int]:
        if self.discard: return {}
        return {"ttWriteRelOffset": self.relOffset}


@NonRVFILine.createSubLineType
class CDPPcTableCollisionLine(NonRVFILine):
    """PC table collision: different PC evicted existing entry at the same index."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel PC table collision"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel PC table collision at idx:\s*\d+ evicted pcHash: ([0-9a-f]+) new pcHash: ([0-9a-f]+)"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPPcTableCollisionLine.dataRegex(line)
        self.evictedPcHash = int(reData[1], 16)
        self.newPcHash     = int(reData[2], 16)

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {"pctCollisions": 1}


@NonRVFILine.createSubLineType
class CDPPcTableUpdateLine(NonRVFILine):
    """PC table confidence updated after a training hit."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel PC table updated"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel PC table updated, idx:\s*\d+ pcHash: ([0-9a-f]+) relOffset:\s*([+-]?\d+) conf:\s*(\d+) ->\s*(\d+)"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPPcTableUpdateLine.dataRegex(line)
        self.pcHash    = int(reData[1], 16)
        self.relOffset = int(reData[2])
        self.oldConf   = int(reData[3])
        self.newConf   = int(reData[4])

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {"pctUpdates": 1}

    def getDistributions(self) -> dict[str, int]:
        if self.discard: return {}
        return {
            "pctUpdateRelOffset": self.relOffset,
            "pctNewConf":         self.newConf,
        }


@NonRVFILine.createSubLineType
class CDPPrefetchDecisionLine(NonRVFILine):
    """Prefetch issued: high-confidence offset selected from PCT entry."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel prefetch decision"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel prefetch decision: pcHash ([0-9a-f]+) relOffset\s*([+-]?\d+) conf\s*(\d+) isNeighbour ([01])"

    # Max cycles to search forward for the matching CDPFilterMissLine
    MAX_DECISION_TO_FILTER_CYCLES = 50

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPPrefetchDecisionLine.dataRegex(line)
        self.pcHash      = int(reData[1], 16)
        self.relOffset   = int(reData[2])
        self.conf        = int(reData[3])
        self.isNeighbour = reData[4] == "1"
        # Set by CDPTrainingHitLine.postProcess (Part 3)
        self.trainingHitLine = None
        self.originCrossPage: bool | None = None
        # Set in self.postProcess (Part 2)
        self.filterMissLine = None

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Link forward to the first unclaimed CDPFilterMissLine.
        # Stop early on a matching-lineAddr filter-HIT (dedup dropped our candidate).
        for ll in after:
            if isinstance(ll, TimestampedLine) and ll.timestamp > self.timestamp + self.MAX_DECISION_TO_FILTER_CYCLES:
                break
            if isinstance(ll, CDPFilterHitLine):
                # Duplicate prefetch dropped; no filter-miss will follow for this decision
                break
            if isinstance(ll, CDPTlbExceptionLine):
                # TLB exception dropped this prefetch before filter
                break
            if isinstance(ll, CDPFilterMissLine) and not ll.decisionClaimed:
                ll.decisionClaimed = True
                ll.originCrossPage = self.originCrossPage
                ll.offsetCategory  = "neighbour" if self.isNeighbour else "inBounds"
                ll.decisionLine    = self   # reverse link so useful-hit can read conf
                self.filterMissLine = ll
                break

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        base = {
            "prefetchDecisions":          1,
            "prefetchDecisionsNeighbour": int(self.isNeighbour),
            "prefetchDecisionsInBounds":  int(not self.isNeighbour),
        }
        fm = self.filterMissLine
        if fm is not None and fm.cRqCreationLine is not None:
            c = fm.cRqCreationLine
            if c.miss:
                verdict = "Useful" if not c.isNeverAccessed else "Useless"
                base[f"cdp{verdict}AtConf{self.conf}"] = 1
        return rt | base

    def getDistributions(self) -> dict[str, int]:
        if self.discard: return {}
        rt: dict[str, int] = {
            "prefetchDecisionRelOffset": self.relOffset,
            "prefetchDecisionConf":      self.conf,
        }
        fm = self.filterMissLine
        if fm is not None and fm.cRqCreationLine is not None:
            c = fm.cRqCreationLine
            if c.miss:
                if not c.isNeverAccessed:
                    rt["cdpConfOfUseful"]  = self.conf
                else:
                    rt["cdpConfOfUseless"] = self.conf
        return rt


@NonRVFILine.createSubLineType
class CDPNoHighConfLine(NonRVFILine):
    """PCT entry existed but no offset met the confidence threshold."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel no high-conf offset"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel no high-conf offset: pcHash ([0-9a-f]+) maxConf\s*(\d+)"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPNoHighConfLine.dataRegex(line)
        self.pcHash   = int(reData[1], 16)
        self.maxConf  = int(reData[2])

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {"pctLookupNoHighConf": 1}

    def getDistributions(self) -> dict[str, int]:
        if self.discard: return {}
        return {
            "noHighConfMaxConf": self.maxConf,
        }


@NonRVFILine.createSubLineType
class CDPTlbExceptionLine(NonRVFILine):
    """TLB translation failed for a prefetch candidate."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel TLB resp: exception"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel TLB resp: exception for vaddr ([0-9a-f]+), dropping prefetch"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPTlbExceptionLine.dataRegex(line)
        self.vaddr = int(reData[1], 16)

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {"tlbExceptions": 1}


@NonRVFILine.createSubLineType
class CDPTlbRespLine(NonRVFILine):
    """TLB translation success: vaddr -> paddr (lineAddr = paddr >> 6)."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel TLB resp: vaddr"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel TLB resp: vaddr ([0-9a-f]+) -> paddr ([0-9a-f]+) lineAddr ([0-9a-f]+) crossPage ([01])"

    # Set of lineAddrs currently known to originate from a neighbour-chain pointer.
    # Consumed by CDPFilterMissLine.postProcess to tag chain-originated prefetches.
    CHAIN_LINEADDR_LOOKOUT: set[int] = set()

    # lineAddr -> crossPage. Populated directly from the authoritative crossPage flag
    # emitted by the BSV TLB-resp log line; consumed by CDPFilterMissLine.
    LINEADDR_TO_CROSSPAGE: dict[int, bool] = {}

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPTlbRespLine.dataRegex(line)
        self.vaddr     = int(reData[1], 16)
        self.paddr     = int(reData[2], 16)
        self.lineAddr  = int(reData[3], 16)
        self.crossPage = reData[4] == "1"

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # If this TLB resp is for a chain-origin vaddr, propagate to lineAddr lookout
        if self.vaddr in CDPNeighbourChainLine.CHAIN_VADDR_LOOKOUT:
            CDPNeighbourChainLine.CHAIN_VADDR_LOOKOUT.discard(self.vaddr)
            CDPTlbRespLine.CHAIN_LINEADDR_LOOKOUT.add(self.lineAddr)

        # Authoritative pageCategory attribution: record crossPage keyed by lineAddr
        # for the downstream CDPFilterMissLine to consume.
        CDPTlbRespLine.LINEADDR_TO_CROSSPAGE[self.lineAddr] = self.crossPage

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {"tlbTranslations": 1}


@NonRVFILine.createSubLineType
class CDPFilterHitLine(NonRVFILine):
    """Prefetch dedup filter blocked a duplicate prefetch."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel filter HIT"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel filter HIT: dropped duplicate prefetch for lineAddr ([0-9a-f]+)"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPFilterHitLine.dataRegex(line)
        self.lineAddr = int(reData[1], 16)

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {"filterDuplicatesDropped": 1}


@NonRVFILine.createSubLineType
class CDPFilterMissLine(NonRVFILine):
    """Prefetch dedup filter passed: new prefetch issued to memory."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel filter MISS"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel filter MISS: issuing prefetch for lineAddr ([0-9a-f]+)"

    # Max cycles to search forward for the matching CRqCreationLine
    MAX_LINK_CYCLES = 200

    # Per-lineAddr lookout: most recent filter-MISS for this lineAddr, consumed
    # by CDPUsefulPrefetchLine to attribute the useful-hit back to the decision
    # that spawned it (without going through the fragile CRqCreation linking).
    LINEADDR_RECENT_FILTER_MISS: "dict[int, 'CDPFilterMissLine']" = {}

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPFilterMissLine.dataRegex(line)
        self.lineAddr = int(reData[1], 16)
        # Set in self.postProcess
        self.cRqCreationLine = None
        # Set by CDPPrefetchDecisionLine.postProcess to prevent double-claim
        self.decisionClaimed = False
        # Propagated from the upstream decision (Part 4.5)
        self.originCrossPage: bool | None = None
        self.offsetCategory: str | None = None   # "inBounds" | "neighbour"
        # Set in self.postProcess (Part 4.5 chain tracking)
        self.chainCategory: str = "direct"       # "direct" | "chain"
        # Reverse link to decision (set by CDPPrefetchDecisionLine.postProcess)
        self.decisionLine: "CDPPrefetchDecisionLine | None" = None

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        # Chain category: was this lineAddr produced by a neighbour-chain prefetch?
        if self.lineAddr in CDPTlbRespLine.CHAIN_LINEADDR_LOOKOUT:
            self.chainCategory = "chain"
            CDPTlbRespLine.CHAIN_LINEADDR_LOOKOUT.discard(self.lineAddr)

        # Page category: fallback to the direct lineAddr map populated by the
        # matching CDPTlbRespLine. Covers the common case where the decision ->
        # filter-miss pcHash chain didn't resolve (most issued prefetches).
        if self.originCrossPage is None:
            crossPage = CDPTlbRespLine.LINEADDR_TO_CROSSPAGE.pop(self.lineAddr, None)
            if crossPage is not None:
                self.originCrossPage = crossPage

        # Register as the most-recent filter MISS for this lineAddr, so a later
        # CDPUsefulPrefetchLine can attribute its useful-hit back to the
        # decision's confidence. This bypasses the fragile cRq-creation linking.
        CDPFilterMissLine.LINEADDR_RECENT_FILTER_MISS[self.lineAddr] = self

        # Link forward to the CRqCreationLine the prefetch triggers
        for ll in after:
            if isinstance(ll, TimestampedLine) and ll.timestamp > self.timestamp + self.MAX_LINK_CYCLES:
                break
            if (isinstance(ll, CRqCreationLine) and ll.isPrefetch
                    and ll.lineAddr == self.lineAddr and not ll.cdpLinked):
                self.cRqCreationLine = ll
                ll.cdpLinked = True
                break

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        base = {"filterPrefetchesIssued": 1}
        c = self.cRqCreationLine
        if c is None:
            return rt | base | {"cdpPrefetchUnlinked": 1}

        useful  = bool(c.miss and not c.isNeverAccessed)
        useless = bool(c.miss and c.isNeverAccessed)
        base |= {
            "cdpPrefetchLinked":   1,
            "cdpPrefetchHitL1":    int(c.hit),
            "cdpPrefetchMissL1":   int(c.miss),
            "cdpPrefetchOwned":    int(c.owned),
            "cdpPrefetchUseful":   int(useful),
            "cdpPrefetchUseless":  int(useless),
            "cdpPrefetchLate":     int(c.isLatePrefetch),
            "cdpPrefetchDisrupt":  int(c.miss and c.disruptedCache),
        }

        # Part 4.5: category-scoped useful/useless totals
        verdict = "Useful" if useful else ("Useless" if useless else None)
        if verdict:
            page = "crossPage" if self.originCrossPage is True else (
                   "samePage"  if self.originCrossPage is False else "unknownPage")
            off  = self.offsetCategory or "unknownOffset"
            ch   = self.chainCategory
            base[f"cdpPref_{page}_{verdict}"]             = 1
            base[f"cdpPref_{off}_{verdict}"]              = 1
            base[f"cdpPref_{ch}_{verdict}"]               = 1
            base[f"cdpPref_{page}_{off}_{ch}_{verdict}"]  = 1
        return rt | base

    def getDistributions(self) -> dict[str, int]:
        if self.discard: return {}
        c = self.cRqCreationLine
        if c is not None and c.prefetchLeadTime is not None:
            return {"cdpPrefetchLeadTime": c.prefetchLeadTime}
        return {}


@NonRVFILine.createSubLineType
class CDPUsefulPrefetchLine(NonRVFILine):
    """Demand hit on a previously prefetched line (cUseful increment)."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel useful prefetch hit"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel useful prefetch hit addr ([0-9a-f]+) cUseful\s*(\d+)"

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPUsefulPrefetchLine.dataRegex(line)
        self.addr    = int(reData[1], 16)
        self.lineAddr = self.addr >> 6
        self.cUseful = int(reData[2])
        # Set in postProcess if we can trace the useful hit back to the decision
        # that issued the matching prefetch.
        self.decisionLine: "CDPPrefetchDecisionLine | None" = None

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return
        # Look up the most-recent filter MISS for this lineAddr and follow its
        # reverse link back to the decision. Pop the entry so a later useful-hit
        # on a re-prefetched line doesn't double-attribute. This gives direct
        # per-conf accuracy WITHOUT the fragile cRq-creation linking chain.
        fm = CDPFilterMissLine.LINEADDR_RECENT_FILTER_MISS.pop(self.lineAddr, None)
        if fm is not None and fm.decisionLine is not None:
            self.decisionLine = fm.decisionLine

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        base = {"usefulPrefetches": 1}
        # Per-confidence accounting via the direct decision link. This lives
        # alongside CDPPrefetchDecisionLine.cdp{Useful,Useless}AtConfN; the
        # "Direct" suffix signals it's computed from the useful-hit emit rather
        # than the cRq-creation linking chain (more robust — see plan.md).
        if self.decisionLine is not None:
            base[f"usefulPrefetchesAtConf{self.decisionLine.conf}_Direct"] = 1
            base[f"usefulPrefetchesNeighbour_Direct"] = int(self.decisionLine.isNeighbour)
            base[f"usefulPrefetchesInBounds_Direct"]  = int(not self.decisionLine.isNeighbour)
        else:
            base["usefulPrefetchesUnattributed_Direct"] = 1
        return rt | base


@NonRVFILine.createSubLineType
class CDPNeighbourChainLine(NonRVFILine):
    """Neighbour-line prefetch returned: attempt to chain a pointer prefetch."""

    _TEST_REGEX = r"^\s*\d+ AlexLog: CDP Rel neighbour chain"
    _DATA_REGEX = r"^\s*(\d+) AlexLog: CDP Rel neighbour chain: word\s*\d+ (?:candidate )?vaddr ([0-9a-f]+) (queued for TLB|failed VPN check, dropping)"

    # Set of vaddrs queued from a neighbour-chain pointer-chase.
    # Consumed by CDPTlbRespLine.postProcess to propagate to CHAIN_LINEADDR_LOOKOUT.
    CHAIN_VADDR_LOOKOUT: set[int] = set()

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CDPNeighbourChainLine.dataRegex(line)
        self.vaddr     = int(reData[1], 16)
        self.vpnFailed = "failed" in reData[2]

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return
        if not self.vpnFailed:
            CDPNeighbourChainLine.CHAIN_VADDR_LOOKOUT.add(self.vaddr)

    def getTotals(self) -> dict[str, int]:
        rt = super().getTotals()
        if self.discard: return rt
        return rt | {
            "neighbourChainAttempts":  1,
            "neighbourChainVpnFailed": int(self.vpnFailed),
        }


class LogParser:

    class openMaybeGZip:
        def __init__(self, filename: str):
            self.filename = filename
        def __enter__(self):
            if self.filename.endswith(".gz") or self.filename.endswith(".gzip"):
                self.fp = gzip.open(self.filename, "rb")
            else:
                self.fp = open(self.filename, "rb")
            return self.fp
        def __exit__(self, *args):
            self.fp.close()

    @staticmethod
    def niceReadChunk(fp: BinaryIO, chunksize=128*1024*1024) -> Iterable[str]:
        leftovers = ""
        while (chunk := fp.read(chunksize).decode("utf-8")):
            lines = chunk.split('\n')
            lines[0] = leftovers + lines[0]
            leftovers = lines.pop()
            yield lines
            del lines
            del chunk

    @staticmethod
    def niceReadLines(fp: BinaryIO, chunksize=128*1024*1024) -> Iterable[str]:
        for chunk in LogParser.niceReadChunk(fp, chunksize):
            for line in chunk:
                yield line.strip()

    def __init__(
        self, 
        log: str, 
        skipLines: int | None = None,
        maxLines: int | None = None, 
        lineTypesToPrune: List[type[LogLine]] = [],
        lineTypesToError: List[type[LogLine]] = [],
        RootLogLine: type[LogLine] = LogLine,
        startWhen: Callable[[LogLine], bool] | None = None,
        stopWhen: Callable[[LogLine], bool] | None = None,
        silent: bool = False
    ) -> None:

        self.log = log
        self.logLines: deque[LogLine] = deque()
        self.lineTypeCounts: dict[type[LogLine], int] = {}
        started = startWhen == None
        started = True

        # Load relevant log lines into self.logLines
        # Skip any line types in lineTypesToPrune
        # Error if any lines in lineTypesToError are found
        with LogParser.openMaybeGZip(log) as fp:
            for line in self.niceReadLines(fp):
                LineType = RootLogLine.deduceLineType(line)              
                # Check lineTypesToPrune (continue on match)
                if LineType in lineTypesToPrune:
                    continue
                # Check lineTypesToError
                if LineType is None or LineType in lineTypesToError:
                    raise ValueError(f"Found LineType {LineType.__name__}, which is either None or in the error list")
                # Skip the line if the user specified skipLines and there are lines remaining to skip
                if skipLines is not None and skipLines > 0:
                    skipLines -= 1
                    continue
                # Instantiate the line type
                try:
                    logLine = LineType(line)
                    # print("LineType instantiated", LineType)
                except Exception as e:
                    raise type(e)(f"Failed to load '{log}': {e}")
                # Check if we need to skip this line because we haven't started
                started = started or startWhen(logLine)
                if not started: continue
                # Save the line to logLines
                # print("Saved line")
                self.logLines.append(logLine)
                self.lineTypeCounts[LineType] = self.lineTypeCounts.get(LineType, 0) + 1
                # Check we haven't recorded maxLines log lines
                if maxLines is not None and len(self.logLines) >= maxLines:
                    break
                # Check if we should stop
                if stopWhen is not None and stopWhen(logLine):
                    break
                # Print a status update
                if not silent and len(self.logLines) % 10000 == 0:
                    print(f"\rLoaded {len(self.logLines)} log lines", end="")

        # Final line counts
        if not silent:
            print(f"\rLoaded {len(self.logLines)} log lines")
            maxLineNameLength = max(len(LineType.__name__) for LineType in self.lineTypeCounts)
            for LineType, count in self.lineTypeCounts.items():
                print(f"\t{(LineType.__name__+':').ljust(maxLineNameLength+1)} {count} instances")

        # Post-process lines
        for process in ["Pre", "Post"]:
            before = deque()
            after  = self.logLines
            while len(after) != 0:
                current = after.popleft()
                if process == "Pre":
                    current.preProcess(before, after)
                else:
                    current.postProcess(before, after)
                before.append(current)
                if not silent and len(before) % 10000 == 0:
                    print(f"\r{process}-processed {len(before)} log lines", end="")
            del after
            self.logLines = before
            if not silent:
                print(f"\r{process}-processed {len(self.logLines)} log lines")
        for i, line in enumerate(self.logLines):
            line.endProcess()
            if not silent and i % 10000 == 0:
                print(f"\rEnd-processed {i} log lines", end="")
        if not silent:
            print(f"\rEnd-processed {len(self.logLines)} log lines")

        # Calculate totals and dists
        self.recalculateTotalsAndDists(silent)

        # Final simulation cycle (last TimestampedLine.timestamp)
        self.finalCycle: int | None = None
        for ll in reversed(self.logLines):
            if isinstance(ll, TimestampedLine):
                self.finalCycle = ll.timestamp
                break



    def recalculateTotalsAndDists(self, silent: bool = False):
        # Calculate totals and distributions
        self.totals = {}
        self.dists  = {}
        for i, ll in enumerate(self.logLines):
            totals = self.totals.setdefault(ll.__class__, {})
            dists  = self.dists.setdefault(ll.__class__, {})
            for k, v in ll.getTotals().items():
                totals[k] = totals.get(k, 0) + v
            for k, v in ll.getDistributions().items():
                if k not in dists:
                    dists[k] = []
                if isinstance(v, list):
                    dists[k].extend(v)
                else:
                    dists[k].append(v)
            if not silent and i % 10000 == 0:
                print(f"\rAccumulated totals and dists for {i} log lines", end="")
        if not silent:
            print(f"\rAccumulated totals and dists for {len(self.logLines)} log lines")
            
        

    def printTotals(self) -> None:
        if self.finalCycle is not None:
            print(f"Final cycle: {self.finalCycle}\n")
        for LineType, totals in self.totals.items():
            if len(totals) != 0:
                print(f"{LineType.__name__} totals:")
                for k, v in totals.items():
                    print(f"\t{k}: {v}")
                print()



    def plotDist(
        self,
        LineType: type[LogLine],
        distName: str,
        figax = None,
        xsAreAddresses = None
    ):
        fig, ax = figax if figax is not None else plt.subplots(figsize=(8, 6))
        data = self.dists[LineType][distName]
        if xsAreAddresses is None:
            xsAreAddresses = min(data) >= 0xc0000000
        xsAddressGranuality = None
        if xsAreAddresses: # Assume addresses
            xsAddressGranuality = min((x & -x).bit_length()-1 for x in data)
            xsAddressGranuality = max(xsAddressGranuality, 2)
            data = [x>>xsAddressGranuality for x in data]
        counts = Counter(data)
        xs = list(counts.keys())
        ys = list(counts.values())
        minxs = min(xs)
        maxxs = max(xs)
        xtickIntervalOpts  = [10**i for i in range(14, 4, -1)] + [50000,20000,10000,5000,2000,1000,100,50,25,20,10,5,2,1] if not xsAreAddresses else [2**x for x in range(32,-1,-1)]
        xtickIdealCount    = 6 if not xsAreAddresses else 12
        xtickExactInterval = max(1, (maxxs-minxs)//xtickIdealCount)
        xtickNiceInterval  = next(i for i in xtickIntervalOpts if i <= xtickExactInterval)
        xtickMin = minxs-(minxs%xtickNiceInterval)
        xtickMax = maxxs+xtickNiceInterval
        ax.bar(xs, ys, color='skyblue', width=1)
        ax.set_title(f"{LineType.__name__}: {distName}")
        ax.set_xlabel("Attribute value")
        ax.set_ylabel("Count")
        if xsAreAddresses:
            ysAvg   = int(statistics.mean(ys))
            ysStdev = int(statistics.stdev(ys))
            ax.set_ylim(0,ysAvg+3*ysStdev)
            ax.set_xticks(
                range(xtickMin, xtickMax, xtickNiceInterval),
                [hex(x << xsAddressGranuality) for x in range(xtickMin, xtickMax, xtickNiceInterval)],
                rotation=90
            )
        else:
            ax.set_xticks(range(xtickMin, xtickMax, xtickNiceInterval))
        if minxs < 0 and maxxs > 0:
            ax.axvline(x=0, linestyle="--", color="black", alpha=0.5)
        ax.grid(axis='y', alpha=0.75)
        return fig,ax

        

    def plotDists(self) -> List:
        figaxs = []
        for LineType, dists in self.dists.items():
            for dataName in dists:
                figaxs.append(self.plotDist(LineType, dataName))
        return figaxs
