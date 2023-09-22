import simpy
import random


# Parameters
simulation_time = 10000  # Total simulation time
ticket_issue_rate_clerk = 1  # Exponential distribution rate for ticket issue at clerk
ticket_issue_rate_machine = 0.7  # Exponential distribution rate for ticket issue at machine
machine_failure_mean = 1000  # Mean of the normal distribution for machine failure rate
machine_failure_std = 200  # Standard deviation of the normal distribution for machine failure rate
repair_time_rate = 1 / 30  # Rate parameter for repair time exponential distribution
train_A_interarrival_rate = 1 / 14  # Rate parameter for train A inter-arrival time exponential distribution
train_B_interarrival_rate = 1 / 25  # Rate parameter for train B inter-arrival time exponential distribution
seats_train = 120  # Total number of seats per train
seat_min = 15  # Minimum seat availability Uniform Random variable
seat_max = 35  # Maximum seat availability Uniform Random variable

# Functions for random events
def generate_interarrival_time(rate):
    return random.expovariate(rate)

def generate_ticket_issue_time(is_machine):
    if is_machine:
        return random.expovariate(ticket_issue_rate_machine)
    else:
        return random.expovariate(ticket_issue_rate_clerk)

def generate_machine_failure_rate():
    return max(0, random.normalvariate(machine_failure_mean, machine_failure_std))

# Simulation
class TrainStation(object):
    def __init__(self, env):
        self.env = env
        # Machine is modelled as pre-emptive resource
        self.machine = simpy.Resource(env, capacity=2)
        self.train_A = simpy.Resource(env, capacity=seats_train)
        self.train_B = simpy.Resource(env, capacity=seats_train)
        self.total_passengers = 0
        self.waiting_times = []             # List to store all waiting times
        self.ticket_purchase_times = []     # List to store all ticket purchase times

    def purchase_ticket(self, passenger):

        yield self.env.timeout(generate_ticket_issue_time(passenger.is_machine))
        self.ticket_purchase_times.append(self.env.now)
        if passenger.is_machine:
            failure_rate = generate_machine_failure_rate()
            yield self.env.timeout(failure_rate)  # Machine failure time
            repair_time = random.expovariate(repair_time_rate)
            yield self.env.timeout(repair_time)  # Machine repair time

    def board_train(self, passenger):
        if passenger.destination == 'A':
            yield self.train_A.request()
        elif passenger.destination == 'B':
            yield self.train_B.request()
        
        seat_availability = random.randint(seat_min, seat_max)
        yield self.env.timeout(seat_availability)
        self.total_passengers += 1
        self.waiting_times.append(self.env.now)
#  The arrival of passengers is independent for both cities
# So coin flip is not needed 

class Passenger(object):
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.is_machine = random.choice([True, False])
        self.destination = random.choice(['A', 'B'])
        env.process(self.run())


    def run(self):
        arrival_time = self.env.now
        yield self.env.process(station.purchase_ticket(self))
        yield self.env.process(station.board_train(self))
        print(f"Passenger {self.name} boarded train {self.destination} at time {self.env.now - arrival_time:.2f}")

# simulation environment
env = simpy.Environment()
station = TrainStation(env)

# Passengers creation
for i in range(num_passengers):
    passenger = Passenger(env, i)

# Run the simulation
env.run()

# Printing the results
average_ticket_purchase_time = sum(station.ticket_purchase_times) / len(station.ticket_purchase_times)
average_waiting_time = sum(station.waiting_times) / station.total_passengers

print(f"Average Ticket Purchase Time for 90% of Passengers: {average_ticket_purchase_time:.2f}")
print(f"Average Waiting Time for All Passengers: {average_waiting_time:.2f}")
