
class HeatStorage:

    def __init__(self, volume: float, dtheta: int):
        self.volume = (volume * 0.997 * 4.19 * dtheta) / 3600
        self.loss = 10.5 * volume / 3600
