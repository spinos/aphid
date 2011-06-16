from foo import people, event
conference = event.Meeting()
conference.add_people(people.Mike())
conference.add_people(people.Charlie())
conference.add_people(people.Julie())
conference.greet()

