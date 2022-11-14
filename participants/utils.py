
class HeatStorage:
    # -> 30 bis 48 Liter (BDEW) hot water per day ~ 50 liter/person and day
    def __init__(self, volume: float, d_theta: int):
        self.volume = (volume * 0.997 * 4.19 * d_theta) / 3600
        self.loss = 10.5 * volume / 3600
        self.V0 = 0