class BipartiteMatching:
	def __init__(self, numOfSDpair, numOfRepeater):
		self.left = numOfSDpair
		self.right = numOfRepeater
		self.matchList = [] # [(SDpair_i, Repeater_k), ...]

	def addEdge(self, SDpair_i, Repeater_k): 
		pass

	def getMatch(self):
		return self.matchList # a pair list [(SDpair_i, Repeater_k), ...]

	def startMatch(self): # Do BipartiteMatching
		pass

if __name__ == "__main__":
	Matching = BipartiteMatching(3, 3) # 3 nodes in left set, 3 node in right set
	Matching.add_edge(1, 2)
	Matching.add_edge(2, 3)
	Matching.add_edge(3, 1)
	Matching.add_edge(1, 3)
	Matching.startMatch()
	Matching.getMatch() # [(1, 2), (2, 3), (3, 1)]