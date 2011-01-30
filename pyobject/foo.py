class foo:
	age = 33
	def __init__(self, parent=None):
		print 'foo is born'
	def getAge(self):
		return self.age
	def add(self, a, b):
		return a + b	
	def log(self, a, b, c):
		return "TaskForce%i member %s retired in %i" % (a, b, c)
		
class loaf:
	def __init__(self, a, b):
		self.a = a
		self.b = b
	def minus(self):
		return self.a - self.b
		
inst_name = 'loaf'
inst_argv = (13, 31)
