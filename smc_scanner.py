"""
SMC Scanner - Fair Value Gap Detection with CRT + TBS Confirmation
Detects FVG patterns using CRT (ChoCh + Retrace + Target) and TBS (Trade Before Support) confirmation
Author: gonexxpara-jpg
Date: 2026-01-05
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    """Direction enumeration"""
    UP = 1
    DOWN = -1


class ConfirmationStatus(Enum):
    """Confirmation status for signals"""
    PENDING = "PENDING"
    CRT_CONFIRMED = "CRT_CONFIRMED"
    TBS_CONFIRMED = "TBS_CONFIRMED"
    FULLY_CONFIRMED = "FULLY_CONFIRMED"
    REJECTED = "REJECTED"


@dataclass
class FVG:
    """Fair Value Gap structure"""
    index: int
    start_index: int
    end_index: int
    direction: Direction
    top: float
    bottom: float
    size: float
    mitigated: bool = False
    mitigation_index: Optional[int] = None
    confirmation_status: ConfirmationStatus = ConfirmationStatus.PENDING
    crt_confirmed: bool = False
    tbs_confirmed: bool = False
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class ChoCh:
    """Change of Character (ChoCh) structure"""
    index: int
    direction: Direction
    price: float
    description: str


@dataclass
class Retrace:
    """Retrace structure"""
    index: int
    start_price: float
    end_price: float
    retrace_percent: float
    depth: float


@dataclass
class SignalSetup:
    """Complete signal setup with CRT and TBS confirmation"""
    fvg: FVG
    choch: ChoCh
    retrace: Retrace
    crt_confirmation: Dict
    tbs_confirmation: Dict
    entry_signal: bool
    confidence_score: float


class FVGScanner:
    """Fair Value Gap Scanner with SMC confirmation systems"""

    def __init__(self, lookback: int = 50, min_fvg_size: float = 0.001):
        """
        Initialize FVG Scanner
        
        Args:
            lookback: Number of candles to analyze
            min_fvg_size: Minimum FVG size as percentage of price
        """
        self.lookback = lookback
        self.min_fvg_size = min_fvg_size
        self.fvgs: List[FVG] = []
        self.chochs: List[ChoCh] = []
        self.signals: List[SignalSetup] = []

    def detect_fvg(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[FVG]:
        """
        Detect Fair Value Gaps (FVG)
        
        FVG Bullish: Low[i] > High[i-2]
        FVG Bearish: High[i] < Low[i-2]
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            
        Returns:
            List of detected FVGs
        """
        fvgs = []
        
        for i in range(2, len(high)):
            # Bullish FVG
            if low[i] > high[i - 2]:
                fvg_size = (low[i] - high[i - 2]) / high[i - 2]
                if fvg_size >= self.min_fvg_size:
                    fvg = FVG(
                        index=i,
                        start_index=i - 2,
                        end_index=i,
                        direction=Direction.UP,
                        top=low[i],
                        bottom=high[i - 2],
                        size=fvg_size
                    )
                    fvgs.append(fvg)
            
            # Bearish FVG
            elif high[i] < low[i - 2]:
                fvg_size = (low[i - 2] - high[i]) / low[i - 2]
                if fvg_size >= self.min_fvg_size:
                    fvg = FVG(
                        index=i,
                        start_index=i - 2,
                        end_index=i,
                        direction=Direction.DOWN,
                        top=low[i - 2],
                        bottom=high[i],
                        size=fvg_size
                    )
                    fvgs.append(fvg)
        
        self.fvgs = fvgs
        return fvgs

    def detect_choch(self, high: np.ndarray, low: np.ndarray, lookback: int = 20) -> List[ChoCh]:
        """
        Detect Change of Character (ChoCh)
        
        Bullish ChoCh: Price creates higher lows (breaks previous lower low)
        Bearish ChoCh: Price creates lower highs (breaks previous higher high)
        
        Args:
            high: High prices array
            low: Low prices array
            lookback: Lookback period for ChoCh detection
            
        Returns:
            List of detected ChoChs
        """
        chochs = []
        
        for i in range(lookback, len(low)):
            # Bullish ChoCh - Break of previous lower low
            prev_min = np.min(low[i - lookback:i - 1])
            if low[i] > prev_min and i > 0:
                if low[i - 1] <= prev_min and low[i] > prev_min:
                    choch = ChoCh(
                        index=i,
                        direction=Direction.UP,
                        price=low[i],
                        description=f"Bullish ChoCh - Break above {prev_min:.6f}"
                    )
                    chochs.append(choch)
            
            # Bearish ChoCh - Break of previous higher high
            prev_max = np.max(high[i - lookback:i - 1])
            if high[i] < prev_max and i > 0:
                if high[i - 1] >= prev_max and high[i] < prev_max:
                    choch = ChoCh(
                        index=i,
                        direction=Direction.DOWN,
                        price=high[i],
                        description=f"Bearish ChoCh - Break below {prev_max:.6f}"
                    )
                    chochs.append(choch)
        
        self.chochs = chochs
        return chochs

    def confirm_crt(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    fvg: FVG, choch: ChoCh, retrace_percent: float = 0.618) -> Dict:
        """
        Confirm FVG using CRT (ChoCh + Retrace + Target)
        
        CRT Confirmation requires:
        1. Change of Character (ChoCh) - Structural break
        2. Retrace - Price retraces to previous level (typically 61.8% Fibonacci)
        3. Target - Third driving leg toward FVG
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            fvg: FVG to confirm
            choch: ChoCh structure
            retrace_percent: Retrace percentage (default 61.8%)
            
        Returns:
            Dictionary with CRT confirmation details
        """
        confirmation = {
            "choch_valid": False,
            "retrace_valid": False,
            "target_valid": False,
            "crt_confirmed": False,
            "details": {}
        }
        
        # 1. Validate ChoCh occurrence after FVG detection
        if choch.index > fvg.index:
            confirmation["choch_valid"] = True
            confirmation["details"]["choch"] = {
                "index": choch.index,
                "direction": choch.direction.name,
                "price": choch.price
            }
        
        # 2. Detect Retrace after ChoCh
        if confirmation["choch_valid"] and choch.index < len(low):
            if fvg.direction == Direction.UP:
                # Bullish FVG - Look for retrace down to 61.8%
                high_after_choch = np.max(high[choch.index:min(choch.index + 20, len(high))])
                low_after_choch = np.min(low[choch.index:min(choch.index + 20, len(low))])
                
                retrace_level = high_after_choch - (high_after_choch - low_after_choch) * retrace_percent
                
                if low_after_choch <= retrace_level:
                    confirmation["retrace_valid"] = True
                    confirmation["details"]["retrace"] = {
                        "level": retrace_level,
                        "reached": True,
                        "depth_percent": retrace_percent
                    }
            
            else:  # Bearish FVG
                # Bearish FVG - Look for retrace up to 61.8%
                high_after_choch = np.max(high[choch.index:min(choch.index + 20, len(high))])
                low_after_choch = np.min(low[choch.index:min(choch.index + 20, len(low))])
                
                retrace_level = low_after_choch + (high_after_choch - low_after_choch) * retrace_percent
                
                if high_after_choch >= retrace_level:
                    confirmation["retrace_valid"] = True
                    confirmation["details"]["retrace"] = {
                        "level": retrace_level,
                        "reached": True,
                        "depth_percent": retrace_percent
                    }
        
        # 3. Validate Target - Third driving leg
        if confirmation["retrace_valid"]:
            retrace_index = choch.index + 10  # Approximate retrace location
            if retrace_index < len(close):
                # Check if price is moving toward FVG (third leg)
                if fvg.direction == Direction.UP:
                    # Price should be moving up toward FVG bottom
                    recent_high = np.max(high[retrace_index:min(retrace_index + 15, len(high))])
                    if recent_high > low_after_choch:
                        confirmation["target_valid"] = True
                        confirmation["details"]["target"] = {
                            "direction": "UP",
                            "target_level": fvg.bottom,
                            "current_direction": "BULLISH"
                        }
                
                else:  # Bearish FVG
                    # Price should be moving down toward FVG top
                    recent_low = np.min(low[retrace_index:min(retrace_index + 15, len(low))])
                    if recent_low < high_after_choch:
                        confirmation["target_valid"] = True
                        confirmation["details"]["target"] = {
                            "direction": "DOWN",
                            "target_level": fvg.top,
                            "current_direction": "BEARISH"
                        }
        
        # CRT is confirmed if all three components are valid
        confirmation["crt_confirmed"] = (
            confirmation["choch_valid"] and
            confirmation["retrace_valid"] and
            confirmation["target_valid"]
        )
        
        return confirmation

    def confirm_tbs(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    fvg: FVG, lookback: int = 20) -> Dict:
        """
        Confirm FVG using TBS (Trade Before Support/Resistance)
        
        TBS Confirmation:
        1. Identify strong support/resistance level
        2. Price trades before breaking through support/resistance
        3. Entry triggered on break with confirmation
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            fvg: FVG to confirm
            lookback: Lookback period for support/resistance
            
        Returns:
            Dictionary with TBS confirmation details
        """
        confirmation = {
            "support_found": False,
            "resistance_found": False,
            "tbs_valid": False,
            "details": {}
        }
        
        if fvg.index >= lookback:
            # Get historical data before FVG
            hist_high = high[max(0, fvg.index - lookback):fvg.index]
            hist_low = low[max(0, fvg.index - lookback):fvg.index]
            
            # For bullish FVG - Find support level (previous lower low)
            if fvg.direction == Direction.UP:
                support_level = np.min(hist_low)
                support_index = np.argmin(hist_low) + max(0, fvg.index - lookback)
                
                # Check if price trades near support before FVG
                near_support = False
                for i in range(max(0, fvg.index - 10), fvg.index):
                    if low[i] <= support_level * 1.001:  # Within 0.1% of support
                        near_support = True
                        break
                
                if near_support:
                    confirmation["support_found"] = True
                    confirmation["tbs_valid"] = True
                    confirmation["details"]["support"] = {
                        "level": support_level,
                        "index": support_index,
                        "touched_before_fvg": True
                    }
            
            # For bearish FVG - Find resistance level (previous higher high)
            else:
                resistance_level = np.max(hist_high)
                resistance_index = np.argmax(hist_high) + max(0, fvg.index - lookback)
                
                # Check if price trades near resistance before FVG
                near_resistance = False
                for i in range(max(0, fvg.index - 10), fvg.index):
                    if high[i] >= resistance_level * 0.999:  # Within 0.1% of resistance
                        near_resistance = True
                        break
                
                if near_resistance:
                    confirmation["resistance_found"] = True
                    confirmation["tbs_valid"] = True
                    confirmation["details"]["resistance"] = {
                        "level": resistance_level,
                        "index": resistance_index,
                        "touched_before_fvg": True
                    }
        
        return confirmation

    def calculate_entry_signal(self, fvg: FVG, crt_confirmation: Dict, tbs_confirmation: Dict) -> Tuple[bool, float]:
        """
        Calculate entry signal based on CRT + TBS confirmation
        
        Args:
            fvg: FVG structure
            crt_confirmation: CRT confirmation dictionary
            tbs_confirmation: TBS confirmation dictionary
            
        Returns:
            Tuple of (signal_valid, confidence_score)
        """
        confidence = 0.0
        
        # CRT confirmation adds 60% confidence
        if crt_confirmation.get("crt_confirmed"):
            confidence += 0.60
        elif crt_confirmation.get("choch_valid"):
            confidence += 0.20
        
        # TBS confirmation adds 40% confidence
        if tbs_confirmation.get("tbs_valid"):
            confidence += 0.40
        
        # Signal is valid if total confidence >= 80%
        signal_valid = confidence >= 0.80
        
        return signal_valid, confidence

    def calculate_risk_management(self, fvg: FVG, close: float, risk_reward_ratio: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate entry, stop loss, and take profit levels
        
        Args:
            fvg: FVG structure
            close: Current close price
            risk_reward_ratio: Risk to reward ratio (default 2:1)
            
        Returns:
            Tuple of (entry_price, stop_loss, take_profit)
        """
        entry_price = None
        stop_loss = None
        take_profit = None
        
        if fvg.direction == Direction.UP:
            # Bullish FVG - Entry at bottom of FVG
            entry_price = fvg.bottom
            stop_loss = fvg.bottom * 0.995  # Below FVG
            
            # Calculate take profit based on FVG size and risk reward
            risk = entry_price - stop_loss
            reward = risk * risk_reward_ratio
            take_profit = entry_price + reward
        
        else:  # Bearish FVG
            # Bearish FVG - Entry at top of FVG
            entry_price = fvg.top
            stop_loss = fvg.top * 1.005  # Above FVG
            
            # Calculate take profit based on FVG size and risk reward
            risk = stop_loss - entry_price
            reward = risk * risk_reward_ratio
            take_profit = entry_price - reward
        
        return entry_price, stop_loss, take_profit

    def scan_signals(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[SignalSetup]:
        """
        Complete scan for FVG signals with CRT + TBS confirmation
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            
        Returns:
            List of confirmed signal setups
        """
        # Step 1: Detect FVGs
        fvgs = self.detect_fvg(high, low, close)
        
        # Step 2: Detect ChoChs
        chochs = self.detect_choch(high, low)
        
        # Step 3: Match FVGs with CRT + TBS confirmation
        signals = []
        
        for fvg in fvgs:
            # Find relevant ChoCh for this FVG
            relevant_chochs = [c for c in chochs if c.index > fvg.index]
            
            if relevant_chochs:
                choch = relevant_chochs[0]  # Use first ChoCh after FVG
                
                # CRT Confirmation
                crt_confirmation = self.confirm_crt(high, low, close, fvg, choch)
                
                # TBS Confirmation
                tbs_confirmation = self.confirm_tbs(high, low, close, fvg)
                
                # Calculate entry signal
                entry_signal, confidence = self.calculate_entry_signal(fvg, crt_confirmation, tbs_confirmation)
                
                # Calculate risk management
                entry_price, stop_loss, take_profit = self.calculate_risk_management(fvg, close[fvg.index])
                
                # Update FVG confirmation status
                if crt_confirmation.get("crt_confirmed") and tbs_confirmation.get("tbs_valid"):
                    fvg.confirmation_status = ConfirmationStatus.FULLY_CONFIRMED
                    fvg.crt_confirmed = True
                    fvg.tbs_confirmed = True
                elif crt_confirmation.get("crt_confirmed"):
                    fvg.confirmation_status = ConfirmationStatus.CRT_CONFIRMED
                    fvg.crt_confirmed = True
                elif tbs_confirmation.get("tbs_valid"):
                    fvg.confirmation_status = ConfirmationStatus.TBS_CONFIRMED
                    fvg.tbs_confirmed = True
                
                fvg.entry_price = entry_price
                fvg.stop_loss = stop_loss
                fvg.take_profit = take_profit
                
                # Create signal setup
                signal = SignalSetup(
                    fvg=fvg,
                    choch=choch,
                    retrace=Retrace(
                        index=choch.index + 10,
                        start_price=close[choch.index],
                        end_price=close[min(choch.index + 10, len(close) - 1)],
                        retrace_percent=0.618,
                        depth=abs(close[choch.index] - close[min(choch.index + 10, len(close) - 1)])
                    ),
                    crt_confirmation=crt_confirmation,
                    tbs_confirmation=tbs_confirmation,
                    entry_signal=entry_signal,
                    confidence_score=confidence
                )
                
                signals.append(signal)
        
        self.signals = signals
        return signals

    def get_summary(self) -> Dict:
        """
        Get summary of detected signals
        
        Returns:
            Dictionary with signal summary
        """
        valid_signals = [s for s in self.signals if s.entry_signal]
        
        return {
            "total_fvgs_detected": len(self.fvgs),
            "total_chochs_detected": len(self.chochs),
            "total_signals_generated": len(self.signals),
            "valid_signals": len(valid_signals),
            "fully_confirmed": sum(1 for s in self.signals if s.fvg.confirmation_status == ConfirmationStatus.FULLY_CONFIRMED),
            "crt_only": sum(1 for s in self.signals if s.fvg.confirmation_status == ConfirmationStatus.CRT_CONFIRMED),
            "tbs_only": sum(1 for s in self.signals if s.fvg.confirmation_status == ConfirmationStatus.TBS_CONFIRMED),
            "average_confidence": np.mean([s.confidence_score for s in self.signals]) if self.signals else 0.0
        }

    def export_signals(self) -> List[Dict]:
        """
        Export signals as dictionaries for JSON/DataFrame conversion
        
        Returns:
            List of signal dictionaries
        """
        exported = []
        
        for signal in self.signals:
            exported.append({
                "fvg_index": signal.fvg.index,
                "fvg_direction": signal.fvg.direction.name,
                "fvg_size_percent": signal.fvg.size * 100,
                "fvg_top": signal.fvg.top,
                "fvg_bottom": signal.fvg.bottom,
                "choch_index": signal.choch.index,
                "choch_direction": signal.choch.direction.name,
                "confirmation_status": signal.fvg.confirmation_status.value,
                "crt_confirmed": signal.fvg.crt_confirmed,
                "tbs_confirmed": signal.fvg.tbs_confirmed,
                "entry_price": signal.fvg.entry_price,
                "stop_loss": signal.fvg.stop_loss,
                "take_profit": signal.fvg.take_profit,
                "entry_signal": signal.entry_signal,
                "confidence_score": signal.confidence_score
            })
        
        return exported


if __name__ == "__main__":
    # Example usage
    print("SMC Scanner initialized successfully")
    print("Features:")
    print("- Fair Value Gap (FVG) Detection")
    print("- Change of Character (ChoCh) Identification")
    print("- CRT Confirmation (ChoCh + Retrace + Target)")
    print("- TBS Confirmation (Trade Before Support)")
    print("- Risk Management Calculations")
    print("- Signal Confidence Scoring")
