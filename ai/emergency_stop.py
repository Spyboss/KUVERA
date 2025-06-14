"""Emergency Stop System - Circuit Breakers and Anomaly Detection
Provides multiple layers of safety mechanisms for autonomous trading
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

class EmergencyStopSystem:
    """Multi-layered emergency stop system for trading bot"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('EmergencyStop')
        
        # Emergency stop triggers
        self.is_emergency_stopped = False
        self.stop_reasons = []
        
        # Circuit breaker parameters
        self.MAX_DAILY_LOSS = config.get('max_daily_loss_pct', 0.03)  # 3%
        self.MAX_CONSECUTIVE_LOSSES = config.get('max_consecutive_losses', 5)
        self.MAX_DRAWDOWN = config.get('max_drawdown_pct', 0.05)  # 5%
        self.VOLATILITY_THRESHOLD = config.get('volatility_threshold', 5.0)  # 5%
        self.FLASH_CRASH_THRESHOLD = config.get('flash_crash_threshold', 0.10)  # 10%
        
        # Tracking variables
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.trade_history = []
        self.price_history = []
        
        # Time-based controls
        self.last_reset_date = datetime.now().date()
        self.emergency_cooldown = 3600  # 1 hour cooldown after emergency stop
        self.last_emergency_time = 0
        
        # Market condition monitoring
        self.market_conditions = {
            'volatility': 0.0,
            'trend_strength': 0.0,
            'volume_anomaly': False,
            'price_gap': 0.0
        }
        
    def update_balance(self, new_balance: float):
        """Update current balance and track peak"""
        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
    
    def add_trade_result(self, pnl: float, trade_type: str):
        """Add trade result for tracking"""
        # Reset daily counters if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.last_reset_date = current_date
            self.logger.info(f"Daily counters reset for {current_date}")
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Add to trade history
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'type': trade_type,
            'daily_pnl': self.daily_pnl
        })
        
        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)
    
    def update_price_data(self, price: float):
        """Update price data for anomaly detection"""
        self.price_history.append({
            'price': price,
            'timestamp': datetime.now()
        })
        
        # Keep only last 100 price points
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        # Update market conditions
        self._update_market_conditions()
    
    def _update_market_conditions(self):
        """Update market condition indicators"""
        try:
            if len(self.price_history) < 10:
                return
            
            prices = [p['price'] for p in self.price_history[-20:]]
            
            # Calculate volatility
            if len(prices) >= 10:
                returns = np.diff(prices) / prices[:-1]
                self.market_conditions['volatility'] = np.std(returns) * 100
            
            # Calculate trend strength
            if len(prices) >= 5:
                short_ma = np.mean(prices[-5:])
                long_ma = np.mean(prices[-10:]) if len(prices) >= 10 else short_ma
                self.market_conditions['trend_strength'] = abs(short_ma - long_ma) / long_ma * 100
            
            # Check for price gaps
            if len(prices) >= 2:
                price_change = (prices[-1] - prices[-2]) / prices[-2] * 100
                self.market_conditions['price_gap'] = abs(price_change)
            
        except Exception as e:
            self.logger.error(f"Error updating market conditions: {e}")
    
    def check_emergency_conditions(self) -> Tuple[bool, List[str]]:
        """Check all emergency stop conditions"""
        if self.is_emergency_stopped:
            return True, self.stop_reasons
        
        emergency_reasons = []
        
        # 1. Daily loss limit
        if self.current_balance > 0:
            daily_loss_pct = abs(self.daily_pnl) / self.current_balance
            if self.daily_pnl < 0 and daily_loss_pct >= self.MAX_DAILY_LOSS:
                emergency_reasons.append(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
        
        # 2. Consecutive losses
        if self.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            emergency_reasons.append(f"Too many consecutive losses: {self.consecutive_losses}")
        
        # 3. Maximum drawdown
        if self.peak_balance > 0 and self.current_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            if drawdown >= self.MAX_DRAWDOWN:
                emergency_reasons.append(f"Maximum drawdown exceeded: {drawdown:.2%}")
        
        # 4. High volatility
        if self.market_conditions['volatility'] >= self.VOLATILITY_THRESHOLD:
            emergency_reasons.append(f"Extreme volatility: {self.market_conditions['volatility']:.2f}%")
        
        # 5. Flash crash detection
        if self.market_conditions['price_gap'] >= self.FLASH_CRASH_THRESHOLD * 100:
            emergency_reasons.append(f"Flash crash detected: {self.market_conditions['price_gap']:.2f}% price gap")
        
        # 6. Check cooldown period
        if time.time() - self.last_emergency_time < self.emergency_cooldown:
            remaining = self.emergency_cooldown - (time.time() - self.last_emergency_time)
            emergency_reasons.append(f"Emergency cooldown active: {remaining/60:.1f} minutes remaining")
        
        if emergency_reasons:
            self.trigger_emergency_stop(emergency_reasons)
            return True, emergency_reasons
        
        return False, []
    
    def trigger_emergency_stop(self, reasons: List[str]):
        """Trigger emergency stop with given reasons"""
        self.is_emergency_stopped = True
        self.stop_reasons = reasons
        self.last_emergency_time = time.time()
        
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {'; '.join(reasons)}")
        
        # Log emergency details
        self.logger.critical(f"Emergency Stop Details:")
        self.logger.critical(f"  Daily P&L: ${self.daily_pnl:.2f}")
        self.logger.critical(f"  Consecutive Losses: {self.consecutive_losses}")
        self.logger.critical(f"  Current Balance: ${self.current_balance:.2f}")
        self.logger.critical(f"  Peak Balance: ${self.peak_balance:.2f}")
        self.logger.critical(f"  Market Volatility: {self.market_conditions['volatility']:.2f}%")
    
    def can_resume_trading(self) -> Tuple[bool, str]:
        """Check if trading can be resumed after emergency stop"""
        if not self.is_emergency_stopped:
            return True, "Not in emergency stop"
        
        # Check cooldown period
        if time.time() - self.last_emergency_time < self.emergency_cooldown:
            remaining = self.emergency_cooldown - (time.time() - self.last_emergency_time)
            return False, f"Cooldown period: {remaining/60:.1f} minutes remaining"
        
        # Check if conditions have improved
        current_emergency, reasons = self.check_emergency_conditions()
        
        # Temporarily disable emergency stop to check conditions
        temp_emergency_state = self.is_emergency_stopped
        self.is_emergency_stopped = False
        
        current_emergency, reasons = self.check_emergency_conditions()
        
        # Restore emergency state
        self.is_emergency_stopped = temp_emergency_state
        
        if not current_emergency:
            return True, "Conditions normalized"
        else:
            return False, f"Conditions still critical: {'; '.join(reasons)}"
    
    def manual_resume(self) -> bool:
        """Manually resume trading (admin override)"""
        if self.is_emergency_stopped:
            self.logger.warning("Manual resume triggered - emergency stop overridden")
            self.is_emergency_stopped = False
            self.stop_reasons = []
            return True
        return False
    
    def get_risk_assessment(self) -> Dict:
        """Get current risk assessment"""
        try:
            # Calculate risk metrics
            risk_score = 0.0
            risk_factors = []
            
            # Daily loss risk
            if self.current_balance > 0 and self.daily_pnl < 0:
                daily_loss_pct = abs(self.daily_pnl) / self.current_balance
                daily_risk = daily_loss_pct / self.MAX_DAILY_LOSS
                risk_score += daily_risk * 0.3
                if daily_risk > 0.5:
                    risk_factors.append(f"High daily loss: {daily_loss_pct:.2%}")
            
            # Consecutive loss risk
            consecutive_risk = self.consecutive_losses / self.MAX_CONSECUTIVE_LOSSES
            risk_score += consecutive_risk * 0.2
            if consecutive_risk > 0.5:
                risk_factors.append(f"Consecutive losses: {self.consecutive_losses}")
            
            # Drawdown risk
            if self.peak_balance > 0 and self.current_balance > 0:
                drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
                drawdown_risk = drawdown / self.MAX_DRAWDOWN
                risk_score += drawdown_risk * 0.3
                if drawdown_risk > 0.5:
                    risk_factors.append(f"High drawdown: {drawdown:.2%}")
            
            # Volatility risk
            volatility_risk = self.market_conditions['volatility'] / self.VOLATILITY_THRESHOLD
            risk_score += volatility_risk * 0.2
            if volatility_risk > 0.5:
                risk_factors.append(f"High volatility: {self.market_conditions['volatility']:.2f}%")
            
            # Normalize risk score
            risk_score = min(1.0, risk_score)
            
            # Determine risk level
            if risk_score < 0.3:
                risk_level = "LOW"
            elif risk_score < 0.6:
                risk_level = "MEDIUM"
            elif risk_score < 0.8:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'emergency_stopped': self.is_emergency_stopped,
                'daily_pnl': self.daily_pnl,
                'consecutive_losses': self.consecutive_losses,
                'market_conditions': self.market_conditions.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk assessment: {e}")
            return {
                'risk_score': 1.0,
                'risk_level': "ERROR",
                'risk_factors': [f"Risk calculation error: {e}"],
                'emergency_stopped': True,
                'daily_pnl': self.daily_pnl,
                'consecutive_losses': self.consecutive_losses,
                'market_conditions': {}
            }
    
    def should_reduce_position_size(self) -> Tuple[bool, float]:
        """Check if position size should be reduced due to risk"""
        risk_assessment = self.get_risk_assessment()
        risk_score = risk_assessment['risk_score']
        
        if risk_score < 0.3:
            return False, 1.0  # Normal position size
        elif risk_score < 0.6:
            return True, 0.7   # Reduce to 70%
        elif risk_score < 0.8:
            return True, 0.5   # Reduce to 50%
        else:
            return True, 0.2   # Reduce to 20%
    
    def get_status_summary(self) -> str:
        """Get formatted status summary"""
        risk_assessment = self.get_risk_assessment()
        
        status = f"""Emergency Stop System Status:
├─ Emergency Stopped: {'YES' if self.is_emergency_stopped else 'NO'}
├─ Risk Level: {risk_assessment['risk_level']} ({risk_assessment['risk_score']:.2f})
├─ Daily P&L: ${self.daily_pnl:.2f}
├─ Consecutive Losses: {self.consecutive_losses}/{self.MAX_CONSECUTIVE_LOSSES}
├─ Market Volatility: {self.market_conditions['volatility']:.2f}%
└─ Active Risk Factors: {len(risk_assessment['risk_factors'])}"""
        
        if risk_assessment['risk_factors']:
            status += "\n   └─ " + "\n   └─ ".join(risk_assessment['risk_factors'])
        
        return status