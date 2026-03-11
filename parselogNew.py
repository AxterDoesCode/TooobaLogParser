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
    _DATA_REGEX = r"^\d+ L1D cRq creation: mshr: (\d+), addr: (0x[0-9a-f]+), vpn: (0x[0-9a-f]+),pcHash: (0x[0-9a-f]+), mshrInUse: \s*(\d+)/\s*(\d+), isPrefetch: ([01]), isRetry: ([01]), reqCs: ([ITSEM]), op: (Ld|St|Lr|Sc|Amo)"

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
        self.vpn          = int(reData[3], 0)
        self.pcHash       = int(reData[4], 0)
        self.mshrUsed     = int(reData[5])
        self.totalMshr    = int(reData[6])
        self.isPrefetch   = int(reData[7] == "1")
        self.isDemand     = not self.isPrefetch
        self.isRetry      = int(reData[8] == "1")
        self.reqCs        = str(reData[9])
        self.op           = str(reData[10])
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
        # if not self.isPrefetch:
        #     rt["demandCapSize"] = self.boundsLength
        # if self.isPrefetch:
        #     rt["prefetchCapSize"] = self.boundsLength
        # if not self.isPrefetch and self.cRqHitLine is not None and self.cRqHitLine.nCap > 0:
        #     rt["demandHasPtrsCapSize"] = self.boundsLength
        # if self.accessFraction:
        #     rt["smallCapAccessFraction"] = self.accessFraction
        return rt
