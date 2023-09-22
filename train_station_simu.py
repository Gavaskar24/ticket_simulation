# Simulating the ticket issue and boarding process 
# Tickets are issued by two machine and a clerk
# machine has a failure and is declared as pre-emptive resource
# The passengers travel towards either city A or city B
# The passengers are independent of each other and their inter arrival times, arrival times of trains, ticket issue rates 
#  all are given below
import simpy
import random
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

simulation_time = 1440  # Total simulation time
ticket_issue_rate_clerk = 1  # Exponential distribution rate for ticket issue at clerk
ticket_issue_rate_machine = 0.7  # Exponential distribution rate for ticket issue at machine
machine_failure_mean = 1000  # Mean of the normal distribution for machine failure rate
machine_failure_std = 200  # Standard deviation of the normal distribution for machine failure rate
repair_time_rate = 1 / 30  # Rate parameter for repair time exponential distribution

# Train A inter-arrival rates follow beta distribution

train_A_beta_a = 0.502
train_A_beta_b = 146.778
train_A_beta_loc = 7.529999999999999e-08
train_A_beta_scale = 48.573

# Train B inter-arrival rates follow weibull min distribution
train_B_wei_shape=1.459
train_B_wei_location=0.0304
train_B_wei_scale=0.09994

seats_train = 120  # Total number of seats per train
seat_min = 15  # Minimum seat availability Uniform Random variable
seat_max = 35  # Maximum seat availability Uniform Random variable

# Arrival of passengers occurs simultaneously for both cities and are to be generated seperately
# Two lists are created to arrange the passengers after they collect their tickets, so as to board them onto their respective trains

passengers_A = []
passengers_B = []

def machine_failure(env):
    while True:
        yield env.timeout(generate_machine_failure_rate())
        repair_time = random.expovariate(repair_time_rate)
        yield env.timeout(repair_time)  # Machine repair time

def generate_interarrival_time_A():
    return random.betavariate(train_A_beta_a, train_A_beta_b) * train_A_beta_scale + train_A_beta_loc

def generate_interarrival_time_B():
    return random.weibullvariate(train_B_wei_shape, train_B_wei_scale) + train_B_wei_location

def generate_ticket_issue_time(is_machine):
    if is_machine:
        return random.expovariate(ticket_issue_rate_machine)
    else:
        return random.expovariate(ticket_issue_rate_clerk)
    
def generate_machine_failure_rate():
    return max(0, random.normalvariate(machine_failure_mean, machine_failure_std))

def generate_destination():
    return random.choice(['A', 'B'])

#  The choice of passenger between machine and clerk is based on queue length at any of them
# If less is the queue length at machine, the passenger chooses machine and vice versa
# And similarly he chooses machine A or B based on the queue length at the respective machines
# If queue length is tied then he choses at random

class TrainStation(object):
    def __init__(self, env):
        self.env = env
        self.total_passengers = 0
        self.waiting_times = []             # List to store all waiting times
        self.ticket_purchase_times = []     # List to store all ticket purchase times

        

def passenger_arrival(env):
    while True:
        yield env.timeout(generate_interarrival_time_A())
        is_machine = random.choice([True, False])
        destination = generate_destination()
        passenger = Passenger(env, is_machine, destination)
        if is_machine:
            if len(machine.queue) < len(clerk.queue):
                machine.put(passenger)
            elif len(machine.queue) > len(clerk.queue):
                clerk.put(passenger)
            else:
                if random.choice([True, False]):
                    machine.put(passenger)
                else:
                    clerk.put(passenger)
        else:
            if destination == 'A':
                passengers_A.append(passenger)
            elif destination == 'B':
                passengers_B.append(passenger)

def passenger_arrival_B(env):
    while True:
        yield env.timeout(generate_interarrival_time_B())
        is_machine = random.choice([True, False])
        destination = generate_destination()
        passenger = Passenger(env, is_machine, destination)
        if is_machine:
            if len(machine.queue) < len(clerk.queue):
                machine.put(passenger)
            elif len(machine.queue) > len(clerk.queue):
                clerk.put(passenger)
            else:
                if random.choice([True, False]):
                    machine.put(passenger)
                else:
                    clerk.put(passenger)
        else:
            if destination == 'A':
                passengers_A.append(passenger)
            elif destination == 'B':
                passengers_B.append(passenger)


class Passenger(object):
    def __init__(self, env, is_machine, destination):
        self.env = env
        self.is_machine = is_machine
        self.destination = destination
        self.ticket_purchase_time = 0
        self.boarding_time = 0
        self.arrival_time = env.now
        env.process(self.run())

    def run(self):
        self.ticket_purchase_time = self.env.now
        yield self.env.process(ticket_issue(self))
        self.boarding_time = self.env.now
        yield self.env.process(board_train(self))
        self.boarding_time = self.env.now - self.boarding_time
        self.ticket_purchase_time = self.boarding_time - self.ticket_purchase_time
        if self.destination == 'A':
            waiting_times_A.append(self.boarding_time - self.arrival_time)
        elif self.destination == 'B':
            waiting_times_B.append(self.boarding_time - self.arrival_time)

def ticket_issue(passenger):
    yield env.timeout(generate_ticket_issue_time(passenger.is_machine))
    if passenger.is_machine:
        failure_rate = generate_machine_failure_rate()
        yield env.timeout(failure_rate)  # Machine failure time
        repair_time = random.expovariate(repair_time_rate)
        yield env.timeout(repair_time)  # Machine repair time

def board_train(passenger):
    if passenger.destination == 'A':
        yield train_A.request()
    elif passenger.destination == 'B':
        yield train_B.request()
    
    seat_availability = random.randint(seat_min, seat_max)
    yield env.timeout(seat_availability)

# Simulation
env = simpy.Environment()
station = TrainStation(env)
machine = simpy.PreemptiveResource(env, capacity=2)
clerk = simpy.Resource(env, capacity=1)
train_A = simpy.Resource(env, capacity=seats_train)
train_B = simpy.Resource(env, capacity=seats_train)
waiting_times_A = []
waiting_times_B = []

env.process(passenger_arrival(env))
env.process(passenger_arrival_B(env))
env.process(machine_failure(env))
env.run(until=simulation_time)

# Get the average waiting time for ticket purchase for both cities
average_waiting_time_A = np.mean(waiting_times_A)
average_waiting_time_B = np.mean(waiting_times_B)

# Get the average of total time spent by passengers till boarding the train for both cities
average_total_time_A = np.mean([waiting_times_A[i] + passengers_A[i].ticket_purchase_time for i in range(len(waiting_times_A))])
average_total_time_B = np.mean([waiting_times_B[i] + passengers_B[i].ticket_purchase_time for i in range(len(waiting_times_B))])

# Get the 90th percentile of average total time spent by passengers till boarding the train for both cities
percentile_90_A = np.percentile([waiting_times_A[i] + passengers_A[i].ticket_purchase_time for i in range(len(waiting_times_A))], 90)
percentile_90_B = np.percentile([waiting_times_B[i] + passengers_B[i].ticket_purchase_time for i in range(len(waiting_times_B))], 90)

# print the results

print(f"Average waiting time for city A: {average_waiting_time_A:.2f}")
print(f"Average waiting time for city B: {average_waiting_time_B:.2f}")

print(f"Average total time for city A: {average_total_time_A:.2f}")
print(f"Average total time for city B: {average_total_time_B:.2f}")

print(f"90th percentile of total time for city A: {percentile_90_A:.2f}")
print(f"90th percentile of total time for city B: {percentile_90_B:.2f}")

# Plotting the histogram of waiting times for both cities
plt.hist(waiting_times_A, bins=20, density=True, label='City A')
plt.hist(waiting_times_B, bins=20, density=True, label='City B')

plt.title("Waiting times for both cities")
plt.xlabel("Waiting times")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plotting the histogram of total times for both cities

plt.hist([waiting_times_A[i] + passengers_A[i].ticket_purchase_time for i in range(len(waiting_times_A))], bins=20, density=True, label='City A')
plt.hist([waiting_times_B[i] + passengers_B[i].ticket_purchase_time for i in range(len(waiting_times_B))], bins=20, density=True, label='City B')

plt.title("Total times for both cities")
plt.xlabel("Total times")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plotting the histogram of total times for both cities

plt.hist([waiting_times_A[i] + passengers_A[i].ticket_purchase_time for i in range(len(waiting_times_A))], bins=20, density=True, label='City A')
plt.hist([waiting_times_B[i] + passengers_B[i].ticket_purchase_time for i in range(len(waiting_times_B))], bins=20, density=True, label='City B')

plt.title("Total times for both cities")
plt.xlabel("Total times")
plt.ylabel("Frequency")
plt.legend()
plt.show()


