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

    _TEST_REGEX = r"^\d+"
    _DATA_REGEX = r"^(\d+)"

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
    _DATA_REGEX = r"^\d+ L1D cRq creation: mshr: (\d+), addr: (0x[0-9a-f]+), boundsVirtBase: (0x[0-9a-f]+), boundsOffset: (0x[0-9a-f]+), boundsLength: (0x[0-9a-f]+), pcHash: (0x[0-9a-f]+), mshrInUse: \s*(\d+)/\s*(\d+), isPrefetch: ([01]), isRetry: ([01]), reqCs: ([ITSEM]), op: (Ld|St|Lr|Sc|Amo)"

    # How many cycles in the future to expect a cRq response
    MAX_CRQ_RESP_CYCLES = 30

    # How many cycles before a prefetch was even issued to look for a miss for this address
    MAX_LATE_PREFETCH_ISSUE_CYCLES = 20

    # To lookout for how much of capabilities we are accessing
    class CapabilityLookout:
        def __init__(self, timestamp, boundsVirtBase, boundsLength):
            self.timestamp = timestamp
            self.base = boundsVirtBase
            self.length = boundsLength
            # Get the number of cache lines that the bounds cover
            self.nCacheLines = 1
            boundsLength = max(0, boundsLength - (64 - (boundsVirtBase & 0b111111)))
            self.nCacheLines += math.ceil(boundsLength / 64)
            # Create a bitmap for accesses
            self.lineAccessed = 0

        # For when the capability is accessed
        def recordAccess(self, offset):
            line = ((self.base + offset) >> 6) - (self.base >> 6)
            self.lineAccessed |= (1 << line)

        # Get the fraction accessed
        def getAccessFraction(self):
            return self.lineAccessed.bit_count() / self.nCacheLines

        def reset(self, timestamp):
            self.timestamp = timestamp
            self.lineAccessed = 0
        def renew(self, timestamp):
            self.timestamp = timestamp
        
    # The lookout dict for capabilities
    MAX_CAP_USAGE_CYCLES = 1000
    CAP_USAGE_LOOKOUT = {}

    def __init__(self, line: str) -> None:
        super().__init__(line)
        reData = CRqCreationLine.dataRegex(line)
        self.mshr         = int(reData[0])
        self.addr         = int(reData[1], 0)
        self.lineAddr     = self.addr >> 6
        self.boundsBase   = int(reData[2], 0)
        self.boundsOffset = int(reData[3], 0)
        self.boundsLength = int(reData[4], 0)
        self.pcHash       = int(reData[5], 0)
        self.mshrUsed     = int(reData[6])
        self.totalMshr    = int(reData[7])
        self.isPrefetch   = int(reData[8] == "1")
        self.isDemand     = not self.isPrefetch
        self.isRetry      = int(reData[9] == "1")
        self.reqCs        = str(reData[10])
        self.op           = str(reData[11])
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
        if self.isPrefetch and self.boundsLength <= 512:
            cap = self.CAP_USAGE_LOOKOUT.get(self.boundsBase)
            if cap is not None:
                if self.timestamp > cap.timestamp + self.MAX_CAP_USAGE_CYCLES:
                    if cap.getAccessFraction() != 0:
                        self.accessFraction = cap.getAccessFraction()
                    cap.reset(self.timestamp)
                else:
                    cap.renew(self.timestamp)
            else:
                self.CAP_USAGE_LOOKOUT[self.boundsBase] = self.CapabilityLookout(
                    self.timestamp, self.boundsBase, self.boundsLength
                )

        # If this is a small capability, get the usage percentage
        if self.isDemand and self.boundsLength <= 512:
            cap = self.CAP_USAGE_LOOKOUT.get(self.boundsBase)
            if cap is not None and self.timestamp < cap.timestamp + self.MAX_CAP_USAGE_CYCLES:
                cap.recordAccess(self.boundsOffset)

    def endProcess(self) -> None:
        super().endProcess()
        self.accessFraction = [
            cap.getAccessFraction() for cap in self.CAP_USAGE_LOOKOUT.values()
        ]
        self.CAP_USAGE_LOOKOUT.clear()

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

            "demandMissToLDS"   : int(not self.isPrefetch and self.miss and self.boundsLength <= 1024),
            "demandMissLoadToLDS" : int(not self.isPrefetch and self.miss and self.op == "Ld" and self.boundsLength <= 1024),
            "demandMissLLToLDS"   : int(not self.isPrefetch and self.miss and not self.cRqResponseLine.hitInLL and self.boundsLength <= 1024),
            "demandMissLLLoadToLDS"   : int(not self.isPrefetch and self.miss and not self.cRqResponseLine.hitInLL and self.op == "Ld" and self.boundsLength <= 1024),

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
        if not self.isPrefetch:
            rt["demandCapSize"] = self.boundsLength
        if self.isPrefetch:
            rt["prefetchCapSize"] = self.boundsLength
        if not self.isPrefetch and self.cRqHitLine is not None and self.cRqHitLine.nCap > 0:
            rt["demandHasPtrsCapSize"] = self.boundsLength
        if self.accessFraction:
            rt["smallCapAccessFraction"] = self.accessFraction
        return rt



@NonRVFILine.createSubLineType
class LLCRqCreationLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ LL cRq creation"
    _DATA_REGEX = r"^\d+ LL cRq creation: mshr: \s*(\d+), addr: (0x[0-9a-f]+), boundsVirtBase: (0x[0-9a-f]+), boundsOffset: (0x[0-9a-f]+), boundsLength: (0x[0-9a-f]+), mshrInUse: \s*(\d+)/\s*(\d+), isPrefetch: ([01]), wasQueued: ([01]), reqCs: ([ITSEM])"
    
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
        self.boundsBase   = int(reData[2], 0)
        self.boundsOffset = int(reData[3], 0)
        self.boundsLength = int(reData[4], 0)
        self.mshrUsed     = int(reData[5])
        self.totalMshr    = int(reData[6])
        self.isPrefetch   = bool(reData[7] == "1")
        self.isDemand     = not self.isPrefetch
        self.isRetry      = bool(reData[8] == "1")
        self.reqCs        = str(reData[9])
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
    _DATA_REGEX = r"^\d+ L1D cRq hit: mshr: \s*(\d+), addr: (0x[0-9a-f]+), cRq is prefetch: ([01]), wasMiss: ([01]), pipeCs: ([ITSEM]), reqCs: ([ITSEM]), saveCs: ([ITSEM]), op: (Ld|St|Lr|Sc|Amo), nCap: (\d), data: (.*)"

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
        self.nCap          = int(reData[8])
        self.lineData      = str(reData[9])
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
            "demandAccessedCap": (self.cRqCreationLine is not None and not self.cRqIsPrefetch and self.op == "Ld" and self.addr % 16 == 0 and self.hitDataLine.tag and self.cRqCreationLine.boundsLength >= 16)
        }

    def getDistributions(self):
        if self.discard:
            return {}
        rt = {}
        if not self.cRqIsPrefetch:
            rt["demandAddr"] = self.addr
        if self.cRqCreationLine is not None and not self.cRqCreationLine.isPrefetch:
            rt["demandNCap"] = self.nCap
        if self.wasMiss:
            rt["missNCap"] = self.nCap
        if self.cRqCreationLine is not None and not self.cRqIsPrefetch and self.op == "Ld" and self.addr % 16 == 0 and self.hitDataLine.tag:
            rt["demandCapSizeForCapLoad"] = self.cRqCreationLine.boundsLength
        if self.cRqCreationLine is not None and not self.cRqIsPrefetch and self.op == "Ld" and self.addr % 16 == 0 and self.hitDataLine.tag and self.cRqCreationLine.boundsLength >= 16:
            rt["demandCapSizeForSensibleCapLoad"] = self.cRqCreationLine.boundsLength
        if self.wasMiss and self.evictionLine is not None:
            rt["evictionCycles"] = self.evictionLine.timestamp - self.timestamp
        if self.wasMiss and self.evictionLine is not None and self.cRqCreationLine is not None and self.cRqCreationLine.boundsLength <= 512:
            rt["smallCapEvictionCycles"] = self.evictionLine.timestamp - self.timestamp
        return rt



@NonRVFILine.createSubLineType
class CRqHitDataLine(NonRVFILine):

    _TEST_REGEX = r"^\d+ L1Bank hit data:"
    _DATA_REGEX = r"^\d+ L1Bank hit data: TaggedData { tag: (True|False), data: <V 'h([0-9a-f]+) 'h([0-9a-f]+)  > }"

    def __init__(self, line: str):
        super().__init__(line)
        reData = CRqHitDataLine.dataRegex(line)
        self.tag           = bool(reData[0] == "True")
        self.data0         = int(reData[1], 16)
        self.data1         = int(reData[2], 16)
    


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
        if self.wasMiss and self.evictionLine is not None and self.cRqCreationLine is not None and self.cRqCreationLine.boundsLength <= 512:
            rt["smallCapEvictionCycles"] = self.evictionLine.timestamp - self.timestamp
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



@NonRVFILine.createSubLineType
class CapPtrCacheDataArrivalLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ Prefetcher reportCacheDataArrival wasMiss"
    _DATA_REGEX = r"^\d+ Prefetcher reportCacheDataArrival wasMiss ([01]) wasPrefetch ([01]) access addr ([0-9a-f]+) boundslen \s*(\d+) offset ([0-9a-f]+) pcHash ([0-9a-f]+) (.*)" 
        
    # We expect that a few cycles, CapPtr should have sent all prefetches to the TLB
    MAX_CAN_PREFETCH_TIME = 12

    # It shouldn't take this long for prefetches to be made
    MAX_PREFETCH_CREATION_DELAY = 50

    def __init__(self, line):
        super().__init__(line)
        reData = CapPtrCacheDataArrivalLine.dataRegex(line)
        self.wasMiss      = bool(reData[0] == "1")
        self.wasPrefetch  = bool(reData[1] == "1")
        self.addr         = int(reData[2], 16)
        self.lineAddr     = self.addr >> 6
        self.boundsLength = int(reData[3])
        self.boundsOffset = int(reData[4], 16)
        self.pcHash       = int(reData[5], 16)
        self.lineData     = str(reData[6])
        # How many actual prefetches it caused and how long that took
        self.nPrefetches = 0
        self.tlbRespLatency    = []
        self.prefetchLatencies = []

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        possiblePrefetches = False
        stopFindingNew = False
        tlbRespToFind = 0
        cRqReqToFind  = 0
        for ll in after:
            if isinstance(ll, TimestampedLine):
                # If there is prefetching opportunity, then we will see CapPtrDataLookupPTableLine
                # in the same cycle.
                if ll.timestamp > self.timestamp and not possiblePrefetches:
                    break
                # We have found the last prefetch creation
                if (stopFindingNew or ll.timestamp > self.timestamp + self.MAX_CAN_PREFETCH_TIME) and tlbRespToFind == 0 and cRqReqToFind == 0:
                    break
                # Give up
                if ll.timestamp > self.timestamp + self.MAX_PREFETCH_CREATION_DELAY:
                    self.warnIf(tlbRespToFind, "remaining TLB resps to find")
                    self.warnIf(cRqReqToFind, "remaining cRq creations to find")
                    break
                # When we see a CapPtrDataLookupPTableLine, we know to look out for CapPtrCanPrefetchLine.
                if isinstance(ll, CapPtrDataLookupPTableLine):
                    if possiblePrefetches:
                        stopFindingNew = True
                    else:
                        possiblePrefetches = True
                        if self.discardIf(ll.addr != self.addr, "Malformed PTable lookup"): return
                        if ll.nCap == 0:
                            break
                # There _should_ be a TLB response to find after seeing this.
                if isinstance(ll, CapPtrCanPrefetchLine) and possiblePrefetches and not stopFindingNew:
                    tlbRespToFind += 1
                # Give up if there is incorrect TLB speculation, as it might cause some requests to be lost.
                # This case happens but is uncommon.
                if isinstance(ll, TLBIncorrectSpeculationLine) and possiblePrefetches:
                    tlbRespToFind = 0
                    stopFindingNew = True
                # We have found a TLB response we are looking for
                if isinstance(ll, CapPtrTLBResponse) and tlbRespToFind and not hasattr(ll, "_attributedToCacheDataArrival"):
                    ll._attributedToCacheDataArrival = True
                    cRqReqToFind += 1
                    tlbRespToFind -= 1
                    self.tlbRespLatency.append(ll.timestamp - self.timestamp)
                # We have found a prefetch (that we haven't attributed to previous data arrival)
                if (isinstance(ll, CRqCreationLine) or isinstance(ll, LLCRqCreationLine)) and ll.isPrefetch and cRqReqToFind and not hasattr(ll, "_attributedToCacheDataArrival"):
                    ll._attributedToCacheDataArrival = True
                    self.nPrefetches += 1
                    cRqReqToFind -= 1
                    self.prefetchLatencies.append(ll.timestamp - self.timestamp)

    def getDistributions(self):
        rt = super().getDistributions()
        if self.discard:
            return rt
        rt["nPrefetches"] = self.nPrefetches
        if self.wasPrefetch:
            rt["chainedPrefetches"] = self.nPrefetches
        rt["triggerCapSize"] = ([self.boundsLength]*self.nPrefetches)
        rt["tlbRespLatency"] = self.tlbRespLatency
        rt["prefetchLatency"] = self.prefetchLatencies
        return rt



@NonRVFILine.createSubLineType
class CapPtrDataLookupPTableLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ Prefetcher reportDataArrival.*caps for prefetch lookups"
    _DATA_REGEX = r"^\d+ Prefetcher reportDataArrival addr ([0-9a-f]+) prefetech ([01]) adding (\d+) caps for prefetch lookups \(clinestartoffset ([0-9a-f]+)\)(.*)" 
        
    def __init__(self, line):
        super().__init__(line)
        reData = CapPtrDataLookupPTableLine.dataRegex(line)
        self.addr             = int(reData[0], 16)
        self.lineAddr         = self.addr >> 6
        self.prefetch         = bool(reData[1] == "1")
        self.nCap             = int(reData[2])
        self.clineStartOffset = int(reData[3], 16)
        self.capData          = str(reData[4])



@NonRVFILine.createSubLineType
class CapPtrCanPrefetchLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ Prefetcher processPtReadForLookup canprefetch"



#@NonRVFILine.createSubLineType
#class CapPtrOutOfBoundsLine(NonRVFILine):
#    
#    _TEST_REGEX = r"^\d+ Prefetcher processPtReadForLookup [0-9a-f]+ out of bounds"



@NonRVFILine.createSubLineType
class TLBIncorrectSpeculationLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ Dtlb incorrectSpeculation killall"



@NonRVFILine.createSubLineType
class CapPtrTLBResponse(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ Prefetcher got TLB response"
    _DATA_REGEX = r"^\d+ Prefetcher got TLB response: TlbRespToPrefetcher { paddr: 'h([0-9a-f]+), .* haveException: (True|False), permsCheckPass: (True|False) }"

    def __init__(self, line):
        super().__init__(line)
        reData = CapPtrTLBResponse.dataRegex(line)
        self.pAddr        = int(reData[0], 16)
        self.exception = bool(reData[1] == "True")
        self.permsCheckPass = bool(reData[2] == "True")



@NonRVFILine.createSubLineType
class CapPtrDataAddTTableEntryLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ Prefetcher reportDataArrival adding training table entry!"
    _DATA_REGEX = r"^\d+ Prefetcher reportDataArrival adding training table entry! access addr ([0-9a-f]+) boundslen \s*(\d+) offset ([0-9a-f]+) prefetch ([01]) pcHash ([0-9a-f]+) ptraddress ([0-9a-f]+) ptrbase ([0-9a-f]+) ptrlength \s*(\d+) tit ([0-9a-f]+) pit ([0-9a-f]+)" 
        
    MAX_TIME_TO_LOOK_FOR_ACCESSES = 500

    def __init__(self, line):
        super().__init__(line)
        reData = CapPtrDataAddTTableEntryLine.dataRegex(line)
        self.addr         = int(reData[0], 16)
        self.lineAddr     = self.addr >> 6
        self.boundsLength = int(reData[1])
        self.boundsOffset = int(reData[2], 16)
        self.pcHash       = int(reData[3], 16)
        self.prefetch     = bool(reData[4] == "1")
        self.ptrAddress   = int(reData[5], 16)
        self.ptrBase      = int(reData[6], 16)
        self.ptrLength    = int(reData[7])
        self.tit          = int(reData[8], 16)
        self.pit          = int(reData[9], 16)

        # Recognise bad pointers to train on
        self.ptrToStackOrAlmighty = self.ptrBase == 0
        self.ptrTooSmall = self.ptrLength < 16
        self.badPtr = self.ptrToStackOrAlmighty or self.ptrTooSmall

        # Set by self.postProcess
        self.offsetsAccessedLd = set()
        self.cacheOffsetsAccessedLd = set()
        self.offsetsAccessedSt = set()
        self.cacheOffsetsAccessedSt = set()
        self.timeBeforeFirstAccess = None
        

    def postProcess(self, before: Iterable[LogLine], after: Iterable[LogLine]) -> None:
        super().postProcess(before, after)
        if self.discard: return

        for ll in after:
            if isinstance(ll, TimestampedLine):
                if isinstance(ll, CRqCreationLine) and ll.isDemand and ll.boundsBase == self.ptrBase and ll.boundsLength == self.ptrLength:
                    if ll.op == "Ld":
                        self.offsetsAccessedLd.add(ll.boundsOffset)
                        self.cacheOffsetsAccessedLd.add(ll.boundsOffset >> 6)
                    if ll.op == "St":
                        self.offsetsAccessedSt.add(ll.boundsOffset)
                        self.cacheOffsetsAccessedSt.add(ll.boundsOffset >> 6)
                    if not self.timeBeforeFirstAccess:
                        self.timeBeforeFirstAccess = ll.timestamp - self.timestamp
                if ll.timestamp >= self.timestamp + self.MAX_TIME_TO_LOOK_FOR_ACCESSES:
                    break

    def getTotals(self):
        rt = super().getTotals()
        return rt if self.discard else rt | {
            "badPtr"      : int(self.badPtr),
            "goodPtr"     : int(not self.badPtr),
            "ptrToStack"  : int(self.ptrToStackOrAlmighty),
            "ptrTooSmall" : int(self.ptrTooSmall)
        }

    def getDistributions(self):
        rt = super().getDistributions()
        if self.discard:
            return rt
        rt["boundsLength"] = self.boundsLength
        rt["ptrLength"] = self.ptrLength
        if not self.badPtr:
            rt["goodOffsetsAccessedLd"]      = len(self.offsetsAccessedLd)
            rt["goodOffsetsAccessedSt"]      = len(self.offsetsAccessedSt)
            rt["goodOffsetsAccessed"]      = len(self.offsetsAccessedLd | self.offsetsAccessedSt)
            rt["goodCacheOffsetsAccessedLd"] = len(self.cacheOffsetsAccessedLd)
            rt["goodCacheOffsetsAccessedSt"] = len(self.cacheOffsetsAccessedSt)
            rt["goodCacheOffsetsAccessed"] = len(self.cacheOffsetsAccessedLd | self.cacheOffsetsAccessedSt)
            if self.timeBeforeFirstAccess:
                rt["goodTimeBeforeAccess"] = self.timeBeforeFirstAccess
        return rt



@NonRVFILine.createSubLineType
class CapPtrPTUpgradeLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ Prefetcher processPtReadUpgrade"
    _DATA_REGEX = r"^\d+ Prefetcher processPtReadUpgrade (hit|miss) pit ([0-9a-f]+) set lastUsedOffset \s*(\d+) to \s*(\d+), changed state to (.*)" 
        
    def __init__(self, line):
        super().__init__(line)
        reData = CapPtrPTUpgradeLine.dataRegex(line)
        self.hit        = reData[0] == "hit"
        self.miss       = reData[0] == "miss"
        self.pit        = int(reData[1], 16)
        self.lastOffset = int(reData[2])
        self.newOffset  = int(reData[3])
        self.state      = reData[4]



@NonRVFILine.createSubLineType
class CapChaserTLBResponseLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ CapChaser (?:L1|LL) TLB response"
    _DATA_REGEX = r"^\d+ CapChaser (L1|LL) TLB response: exception: ([01]), perms: ([01]), confidence: ([01]{7}), (?:l1Conf: ([01]), )?l2Conf: ([01]), (?:depth: \s*(\d+))?"

    def __init__(self, line):
        super().__init__(line)
        reData = CapChaserTLBResponseLine.dataRegex(line)
        self.origin     = str(reData[0])
        self.exception  = bool(reData[1] == "1")
        self.perms      = bool(reData[2] == "1")
        self.confidence = int(reData[3], 2)
        self.l1Conf     = bool(reData[4] == 1) if reData[4] else None 
        self.l2Conf     = bool(reData[5] == 1) if reData[5] else None 
        self.depth      = int(reData[6]) if reData[6] else 0

    def getDistributions(self):
        rt = super().getDistributions()
        if self.discard:
            return rt
        rt["depth"] = self.depth
        return rt
    


@NonRVFILine.createSubLineType
class CapChaserBroadcastLine(NonRVFILine):
    
    _TEST_REGEX = r"^\d+ CapChaser L1 prepared broadcast"
    _DATA_REGEX = r"^\d+ CapChaser L1 prepared broadcast: ptIdxTag: (0x[0-9a-f]+), ptWay: \s*(\d+), confidence: ([01]{7}), bestOffset: \s*(\d+)"

    def __init__(self, line):
        super().__init__(line)
        reData = CapChaserBroadcastLine.dataRegex(line)
        self.ptIdxTag   = int(reData[0], 0)
        self.ptWay      = int(reData[1])
        self.confidence = int(reData[2], 2)
        self.bestOffset = int(reData[3])
    



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
                except Exception as e:
                    raise type(e)(f"Failed to load '{log}': {e}")
                # Check if we need to skip this line because we haven't started
                started = started or startWhen(logLine)
                if not started: continue
                # Save the line to logLines
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
