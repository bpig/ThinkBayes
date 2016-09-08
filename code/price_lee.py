# -*- coding: utf-8 -*-
# __author__ = "shuai.li(286287737@qq.com)"
# __date__ = "2016/9/8"
import csv
import numpy as np

from thinkbayes import *

class History:
    def __init__(self):
        record = self.readData("showcases.2011.csv")
        record += self.readData("showcases.2012.csv")
        self.p1, self.p2, g1, g2, self.d1, self.d2 = zip(*record)
    
    @staticmethod
    def readData(filename):
        fp = open(filename)
        reader = csv.reader(fp)
        ans = []
        for t in reader:
            try:
                data = [int(_) for _ in t[1:]]
                ans += [data]
            except ValueError:
                pass
        return zip(*ans)
    
    def getPerson1Data(self):
        return self.p1, self.d1
    
    def getPerson2Data(self):
        return self.p2, self.d2

class Price(Suite):
    def __init__(self, pmf, player, name=""):
        Suite.__init__(self, pmf, name)
        self.player = player
    
    def Likelihood(self, data, hypo):
        price = hypo
        diff = price - data
        return self.player.diffDensity(diff)

class Player(Pmf):
    n = 101
    xs = np.linspace(0, 75000, n)
    
    def __init__(self, price, diff):
        self.pricePdf = EstimatedPdf(price)
        self.pricePmf = self.pricePdf.MakePmf(self.xs)
        self.diffCdf = MakeCdfFromList(diff)
        mu = 0
        sigma = np.std(diff)
        self.diffPdf = GaussianPdf(mu, sigma)
    
    def diffDensity(self, diff):
        return self.diffPdf.Density(diff)
    
    def probOverBid(self):
        return self.diffCdf.Prob(-1)
    
    def probWorseThan(self, diff):
        return 1 - self.diffCdf.Prob(diff)
    
    def update(self, guess):
        self.prior = Price(self.pricePmf, self, "prior")
        self.posterior = self.prior.Copy("posterior")
        self.posterior.Update(guess)

class Gain:
    def __init__(self, player, opponent):
        self.player = player
        self.oppo = opponent
    
    def expects(self):
        bids = self.player.xs
        gains = [self.expect(bid) for bid in bids]
        return bids, gains
    
    def expect(self, bid):
        suite = self.player.posterior
        total = 0
        for price, prob in sorted(suite.Items()):
            gain = self.gain(price, bid)
            total += gain * prob
        return total
    
    def gain(self, price, bid):
        if bid > price:
            return 0
        diff = price - bid
        prob = self.probWin(diff)
        if diff <= 250:
            return 2 * price * prob
        else:
            return price * prob
    
    def probWin(self, diff):
        return self.oppo.probOverBid() + self.oppo.probWorseThan(diff)

if __name__ == '__main__':
    ht = History()
    player1 = Player(*ht.getPerson1Data())
    player2 = Player(*ht.getPerson2Data())
    
    player1.update(20000)
    player2.update(40000)

    gain1 = Gain(player1, player2)
    gain2 = Gain(player2, player1)

    bids, gains = gain1.expects()
    print "player1", max(zip(gains, bids))

    bids, gains = gain2.expects()
    print "player2", max(zip(gains, bids))
