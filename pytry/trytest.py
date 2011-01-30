try:
	import xyz
	print 'xyz is found'
except ImportError:
	print 'no xyz module!'
	
try:
	print a
except NameError:
	print 'a not defined!'
	
