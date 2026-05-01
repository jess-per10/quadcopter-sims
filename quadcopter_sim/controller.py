import numpy as np

class PIDController:
    """
    A reusable PID controller for a single axis. 
    Create one instance per axis if you want to control. 
    """
    def __init__(self, kp, ki, kd, output_limit=None): # output limit - absolute max absolute output value (anti-windup)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

        #internal state 
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_step = True
        
    def update(self, setpoint, measurement, dt):
        """
        Compute PID output. 

        setpoint - where you want to be
        measurement - where you are now
        dt - time since last update (s)

        returns control output (e.g. desired motor speed)
        """
        # Error - how far from target
        error = setpoint - measurement

        # Integral - accumulated error over time
        self.integral += error * dt

        # Derivative - rate of change of error
        # On first step we have no previous error so derviative is zero
        if self.first_step:
            derivative = 0.0
            self.first_step = False
        else:
            derivative = (error - self.prev_error) / dt 

        # PID output
        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)
        
        # After computing output, also clamp the integral
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)
            # Also prevent integral from growing beyond what's useful
            self.integral = np.clip(self.integral, 
                            -self.output_limit / (self.ki + 1e-6),
                            self.output_limit / (self.ki + 1e-6))
        
        # Store error for next derivative calculation
        self.prev_error = error

        return output
    
    def reset(self):
        """ Reset internal state - call this between simulation runs. """
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_step = True
