'''###################################################################################################################
# Function name :
# Function type :
# Input values  :
# Output values :
# Comment       :
###################################################################################################################'''
class LowPassFilter(object):
    '''###################################################################################################################
    # Function name :
    # Function type : class member
    # Input values  :
    # Output values :
    # Comment       :
    ###################################################################################################################'''
    def __init__(self, tau, ts):
        self.a = 1. / (tau / ts + 1.)
        self.b = tau / ts / (tau / ts + 1.);

        self.last_val = 0.
        self.ready = False
    '''###################################################################################################################
    # Function name :
    # Function type : class member
    # Input values  :
    # Output values :
    # Comment       :
    ###################################################################################################################'''
    def get(self):
        return self.last_val
    '''###################################################################################################################
    # Function name :
    # Function type : class member
    # Input values  :
    # Output values :
    # Comment       :
    ###################################################################################################################'''
    def filt(self, val):
        if self.ready:
            val = self.a * val + self.b * self.last_val
        else:
            self.ready = True

        self.last_val = val
        return val
