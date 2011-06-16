class Meeting(object):
    def __init__(self):
        super(Meeting, self).__init__()
        self.participant = []
        
    def add_people(self, someone):
        self.participant.append(someone)
        
    def greet(self):
        for someone in self.participant:
            someone.greet()
