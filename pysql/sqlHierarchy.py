import sqlite3
import os

class sqlHierarchy:
    def __init__(self, fileName):
        self.dataName = fileName
        if not os.path.isfile(fileName):
            self.createDataBase()
            
    def createDataBase(self):
        conn = sqlite3.connect(self.dataName)
        cur = conn.cursor()
        cur.execute("""create table stocks (name text, id integer, lft integer, rgt integer)""")
        for t in [('animal',0,1,2),
             ]:
             cur.execute('insert into stocks values (?,?,?,?)', t)
        conn.commit()
        cur.close()
        
    def addLeaf(self, parentName, leafName):
        conn = sqlite3.connect(self.dataName)
        cur = conn.cursor()
        cur.execute('select rgt from stocks where id=0')
        
        leafId = cur.fetchone()[0]
            
        leafId /= 2  
        
        cur.execute('select rgt from stocks where name="%s"' % parentName)
        
        end = cur.fetchone()[0]
            
        
        cur.execute('update stocks set lft=lft+2 where lft>%d' % (end-1))
        cur.execute('update stocks set rgt=rgt+2 where rgt>%d' % (end-1))
        cur.execute('insert into stocks values ("%s",%d,%d,%d)' % (leafName, leafId, end, end+1))
        
        
        conn.commit()
        cur.close()
            
    def displayPathTo(self, branchName):
        conn = sqlite3.connect(self.dataName)
        cur = conn.cursor()
        cur.execute('select lft, rgt from stocks where name="%s"' % branchName)
        col = cur.fetchone()
        
        cur.execute('select * from stocks where lft<%d and rgt>%d order by lft' % (col[0], col[1]))

        for row in cur:
            print "%d %s %d\n|" % (row[2], row[0], row[3])
            
        print '%d %s %d'% (col[0], branchName, col[1])
        
        cur.close()
            
    def displayTree(self, branchName):
        conn = sqlite3.connect(self.dataName)
        cur = conn.cursor()
        cur.execute('select lft, rgt from stocks where name="%s"' % branchName)
        col = cur.fetchone()
        print '%d %s %d'% (col[0], branchName, col[1])
        
        cur.execute('select * from stocks where lft>%d and rgt<%d order by lft' % (col[0], col[1]))
        
        right_pre = 0
        level = 0
        level_end = []
        for row in cur:
            if row[2] != right_pre + 1:
                level_end.append( right_pre )
                level += 1
              
            for end in level_end:
                if row[2] == end + 1:
                    level -= 1
   
            margin = '-'*level
            print margin + "%d %s %d" % (row[2], row[0], row[3])
            
                
            for end in level_end:
                if row[3] == end - 1:
                    level -= 1
                
            right_pre = row[3]

        cur.close()
