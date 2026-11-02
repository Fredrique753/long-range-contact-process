# ============================================================
# DATA STRUCTURES AND OBJECT-ORIENTED PROGRAMMING
# LONG-RANGE CONTACT PROCESS RESEARCH PROJECT
# ============================================================
#
# This script demonstrates:
#
#   1. Core Python data structures:
#        - List
#        - Tuple
#        - Dictionary
#        - Set
#
#   2. Object-oriented programming:
#        - Classes and objects
#        - Encapsulation
#        - Inheritance
#        - Polymorphism
#        - Method overriding
#
#   3. Application to the research project:
#        - Lattice sites
#        - Long-range Contact Process simulation
#        - Machine-learning surrogate models
#
# ============================================================


import random
import math
from collections import defaultdict


# ============================================================
# SECTION 1: CORE DATA STRUCTURES
# ============================================================

# ------------------------------------------------------------
# 1.1 List
# ------------------------------------------------------------
# A list stores an ordered collection of items.
# In this project, a list can store the coordinates of
# currently infected sites.

infected_sites = [
    (0, 0),
    (1, 0),
    (0, 1)
]

print("Initial infected sites:")
print(infected_sites)


# Add a newly infected site

infected_sites.append((2, 1))

print("\nAfter infection spreads to a new site:")
print(infected_sites)


# Remove a site after recovery

infected_sites.remove((1, 0))

print("\nAfter one site recovers:")
print(infected_sites)


# ------------------------------------------------------------
# 1.2 Tuple
# ------------------------------------------------------------
# A tuple is an immutable ordered collection.
# It is appropriate for lattice coordinates because a site
# coordinate should not change after it is created.

site_coordinate = (3, -2)

print("\nLattice coordinate stored as a tuple:")
print(site_coordinate)

print("x-coordinate:", site_coordinate[0])
print("y-coordinate:", site_coordinate[1])


# ------------------------------------------------------------
# 1.3 Dictionary
# ------------------------------------------------------------
# A dictionary stores key-value pairs.
# In this project, it can store model parameters or the
# current state of each lattice site.

model_parameters = {
    "lambda": 1.5,
    "alpha": 3.5,
    "recovery_rate": 1.0,
    "t_max": 20
}

print("\nModel parameters stored in a dictionary:")
print(model_parameters)

print("Infection parameter:", model_parameters["lambda"])
print("Long-range exponent:", model_parameters["alpha"])


# A dictionary can also represent the state of a small lattice.

lattice_state = {
    (0, 0): 1,
    (1, 0): 0,
    (0, 1): 1,
    (1, 1): 0
}

print("\nLattice state:")
print(lattice_state)


# ------------------------------------------------------------
# 1.4 Set
# ------------------------------------------------------------
# A set stores unique items and is useful for recording the
# cumulative set of sites that have ever been infected.

ever_infected_sites = {
    (0, 0),
    (1, 0),
    (0, 1)
}

print("\nSites that have ever been infected:")
print(ever_infected_sites)


# Adding a site that is already present does not create a duplicate

ever_infected_sites.add((0, 0))

print("\nAfter attempting to add an existing site:")
print(ever_infected_sites)


# Adding a new site

ever_infected_sites.add((2, 2))

print("\nAfter a new site becomes infected:")
print(ever_infected_sites)


# ============================================================
# SECTION 2: LATTICE SITE CLASS
# ============================================================

class LatticeSite:
    """
    Represents a single site in the two-dimensional lattice.

    Each site has:
        - An x-coordinate
        - A y-coordinate
        - A state: 0 for vacant, 1 for occupied/infected
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = 0
        self.ever_infected = False

    @property
    def coordinate(self):
        """
        Return the site coordinate as a tuple.
        """
        return (self.x, self.y)

    def infect(self):
        """
        Change the site state from vacant to occupied.
        """
        self.state = 1
        self.ever_infected = True

    def recover(self):
        """
        Change the site state from occupied to vacant.
        """
        self.state = 0

    def is_infected(self):
        """
        Return True if the site is currently occupied.
        """
        return self.state == 1

    def __repr__(self):
        return (
            f"LatticeSite(coordinate={self.coordinate}, "
            f"state={self.state})"
        )


# Demonstrate the LatticeSite class

site_a = LatticeSite(0, 0)
site_b = LatticeSite(1, 1)

print("\n============================================================")
print("LATTICE SITE OBJECTS")
print("============================================================")

print(site_a)
print(site_b)

site_a.infect()

print("\nAfter site A becomes infected:")
print(site_a)

site_a.recover()

print("\nAfter site A recovers:")
print(site_a)

print("Site A was ever infected:", site_a.ever_infected)


# ============================================================
# SECTION 3: LONG-RANGE CONTACT PROCESS CLASS
# ============================================================

class LongRangeContactProcess:
    """
    Simulates a simplified long-range Contact Process on Z^2.

    The process begins with one infected site at the origin.

    At each simulation step:
        - An occupied site may recover.
        - An occupied site may attempt to infect another site.
        - The infection probability decreases with distance.
    """

    def __init__(
        self,
        infection_rate,
        alpha,
        recovery_rate=1.0,
        max_sites=1000
    ):
        """
        Initialise the simulation.

        Parameters
        ----------
        infection_rate : float
            Base infection-attempt rate per occupied site.

        alpha : float
            Long-range exponent controlling distance decay.

        recovery_rate : float
            Rate at which an occupied site recovers.

        max_sites : int
            Maximum number of sites that may become infected.
        """

        self.infection_rate = infection_rate
        self.alpha = alpha
        self.recovery_rate = recovery_rate
        self.max_sites = max_sites

        # Currently infected sites stored in a list
        self.currently_infected = [LatticeSite(0, 0)]

        # Cumulative infected sites stored in a set
        self.ever_infected = {(0, 0)}

        # Simulation history stored in a list of dictionaries
        self.history = []

        self.time = 0.0

        self._record_history()

    def _record_history(self):
        """
        Record the current simulation state.
        """

        history_entry = {
            "time": self.time,
            "currently_infected": len(self.currently_infected),
            "ever_infected": len(self.ever_infected)
        }

        self.history.append(history_entry)

    def _generate_displacement(self):
        """
        Generate a long-range displacement.

        The radial distance has a heavy-tailed distribution.

        Returns
        -------
        tuple
            Integer displacement (dx, dy).
        """

        u = random.random()

        r = (1 - u) ** (-1 / (self.alpha - 2))

        theta = random.uniform(0, 2 * math.pi)

        dx = int(round(r * math.cos(theta)))
        dy = int(round(r * math.sin(theta)))

        if dx == 0 and dy == 0:
            return self._generate_displacement()

        return (dx, dy)

    def _select_event(self):
        """
        Select whether the next event is recovery or infection.

        Returns
        -------
        str
            Either "recovery" or "infection".
        """

        total_rate = (
            self.recovery_rate * len(self.currently_infected)
            +
            self.infection_rate * len(self.currently_infected)
        )

        recovery_probability = (
            self.recovery_rate * len(self.currently_infected)
            /
            total_rate
        )

        if random.random() < recovery_probability:
            return "recovery"

        return "infection"

    def _perform_recovery(self):
        """
        Select an occupied site and make it vacant.
        """

        site = random.choice(self.currently_infected)

        site.recover()

        self.currently_infected.remove(site)

    def _perform_infection(self):
        """
        Select an occupied site and attempt to infect a new site.
        """

        source_site = random.choice(self.currently_infected)

        dx, dy = self._generate_displacement()

        target_coordinate = (
            source_site.x + dx,
            source_site.y + dy
        )

        # If the target is already infected, the attempt fails

        for site in self.currently_infected:
            if site.coordinate == target_coordinate:
                return

        # Stop if the maximum number of sites has been reached

        if len(self.ever_infected) >= self.max_sites:
            return

        # Create and infect the new site

        new_site = LatticeSite(
            target_coordinate[0],
            target_coordinate[1]
        )

        new_site.infect()

        self.currently_infected.append(new_site)

        self.ever_infected.add(target_coordinate)

    def run(self, t_max, record_every=1):
        """
        Run the simulation until the time horizon is reached,
        the process becomes extinct, or the site limit is reached.

        Parameters
        ----------
        t_max : float
            Maximum simulation time.

        record_every : int
            Number of events between history records.
        """

        event_count = 0

        while (
            self.time < t_max
            and len(self.currently_infected) > 0
            and len(self.ever_infected) < self.max_sites
        ):

            event_type = self._select_event()

            if event_type == "recovery":
                self._perform_recovery()
            else:
                self._perform_infection()

            self.time += random.expovariate(
                self.recovery_rate + self.infection_rate
            )

            event_count += 1

            if event_count % record_every == 0:
                self._record_history()

        self._record_history()

    @property
    def is_extinct(self):
        """
        Return True if no sites are currently infected.
        """

        return len(self.currently_infected) == 0

    def summary(self):
        """
        Return a summary of the simulation.
        """

        return {
            "infection_rate": self.infection_rate,
            "alpha": self.alpha,
            "final_time": self.time,
            "currently_infected": len(self.currently_infected),
            "ever_infected": len(self.ever_infected),
            "extinct": self.is_extinct
        }


# Demonstrate the Contact Process class

print("\n============================================================")
print("LONG-RANGE CONTACT PROCESS SIMULATION")
print("============================================================")

random.seed(123)

contact_process = LongRangeContactProcess(
    infection_rate=2.0,
    alpha=3.5,
    recovery_rate=1.0,
    max_sites=200
)

contact_process.run(
    t_max=5,
    record_every=10
)

print("\nSimulation summary:")
print(contact_process.summary())

print("\nFirst five history records:")
print(contact_process.history[:5])


# ============================================================
# SECTION 4: INHERITANCE AND POLYMORPHISM
# ============================================================

class SpreadModel:
    """
    Base class representing a general spatial spreading model.
    """

    def __init__(self, name):
        self.name = name

    def describe(self):
        """
        Return a description of the model.
        """

        return f"This is a general spreading model called {self.name}."

    def predict_growth(self):
        """
        Return a placeholder growth prediction.
        """

        return "Growth prediction not defined."


class ContactProcessModel(SpreadModel):
    """
    Represents the long-range Contact Process model.
    """

    def __init__(self, infection_rate, alpha):
        super().__init__("Long-Range Contact Process")

        self.infection_rate = infection_rate
        self.alpha = alpha

    def describe(self):
        """
        Override the base-class description.
        """

        return (
            f"{self.name} with infection rate "
            f"{self.infection_rate} and alpha {self.alpha}."
        )

    def predict_growth(self):
        """
        Return a simple growth prediction.
        """

        if self.infection_rate > 2:
            return "The infection is likely to persist."

        return "The infection is likely to become extinct."


class NeuralNetworkSurrogate(SpreadModel):
    """
    Represents a neural-network surrogate model.
    """

    def __init__(self, architecture):
        super().__init__("Neural-Network Surrogate")

        self.architecture = architecture

    def describe(self):
        """
        Override the base-class description.
        """

        return (
            f"{self.name} with architecture {self.architecture}."
        )

    def predict_growth(self):
        """
        Return a surrogate-model prediction.
        """

        return (
            "The surrogate predicts the boundary coordinate y "
            "from lambda, alpha and x."
        )


# Demonstrate inheritance and polymorphism

print("\n============================================================")
print("INHERITANCE AND POLYMORPHISM")
print("============================================================")

models = [
    ContactProcessModel(
        infection_rate=2.5,
        alpha=3.5
    ),
    NeuralNetworkSurrogate(
        architecture="64-32-16-1"
    )
]

for model in models:
    print("\n" + model.describe())
    print(model.predict_growth())


# ============================================================
# SECTION 5: ENCAPSULATION
# ============================================================

class BoundaryDataset:
    """
    Stores and manages boundary-coordinate data.

    The internal dataset is private and can only be accessed
    through controlled methods.
    """

    def __init__(self):
        self.__records = []

    def add_record(self, record):
        """
        Add a boundary record.

        Parameters
        ----------
        record : dict
            Dictionary containing lambda, alpha, x and y.
        """

        required_keys = {"lambda", "alpha", "x", "y"}

        if not required_keys.issubset(record.keys()):
            raise ValueError(
                "Record must contain lambda, alpha, x and y."
            )

        self.__records.append(record)

    def get_records(self):
        """
        Return a copy of the stored records.
        """

        return self.__records.copy()

    def number_of_records(self):
        """
        Return the number of stored records.
        """

        return len(self.__records)

    def average_y(self):
        """
        Return the mean boundary coordinate y.
        """

        if len(self.__records) == 0:
            return None

        return sum(
            record["y"] for record in self.__records
        ) / len(self.__records)


# Demonstrate encapsulation

print("\n============================================================")
print("ENCAPSULATION")
print("============================================================")

boundary_dataset = BoundaryDataset()

boundary_dataset.add_record(
    {
        "lambda": 0.4,
        "alpha": 3.5,
        "x": 0,
        "y": 10.2
    }
)

boundary_dataset.add_record(
    {
        "lambda": 0.4,
        "alpha": 3.5,
        "x": 1,
        "y": 11.8
    }
)

print("Number of records:", boundary_dataset.number_of_records())
print("Average boundary y:", boundary_dataset.average_y())

print("\nStored records:")
print(boundary_dataset.get_records())


# ============================================================
# SECTION 6: DICTIONARY OF LISTS FOR GROUPING DATA
# ============================================================

print("\n============================================================")
print("GROUPING DATA USING A DICTIONARY OF LISTS")
print("============================================================")

boundary_records = [
    {"lambda": 0.2, "alpha": 3.5, "x": 0, "y": 4.1},
    {"lambda": 0.2, "alpha": 3.5, "x": 1, "y": 5.3},
    {"lambda": 0.4, "alpha": 3.5, "x": 0, "y": 12.7},
    {"lambda": 0.4, "alpha": 3.5, "x": 1, "y": 14.2},
    {"lambda": 0.6, "alpha": 3.5, "x": 0, "y": 25.6},
    {"lambda": 0.6, "alpha": 3.5, "x": 1, "y": 27.9}
]

records_by_lambda = defaultdict(list)

for record in boundary_records:
    records_by_lambda[record["lambda"]].append(record)

for infection_rate, records in records_by_lambda.items():
    print(
        f"lambda = {infection_rate}: "
        f"{len(records)} boundary records"
    )


# ============================================================
# SECTION 7: FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("DATA STRUCTURES AND OOP DEMONSTRATION COMPLETED")
print("============================================================")

print(
    """
This script demonstrated:

1. Lists:
   Used to store currently infected sites and simulation history.

2. Tuples:
   Used to represent immutable lattice coordinates.

3. Dictionaries:
   Used to store model parameters, simulation states and
   grouped boundary records.

4. Sets:
   Used to store the cumulative set of sites that have ever
   been infected.

5. Classes and objects:
   LatticeSite, LongRangeContactProcess, SpreadModel,
   ContactProcessModel, NeuralNetworkSurrogate and
   BoundaryDataset.

6. Encapsulation:
   BoundaryDataset hides its internal records and provides
   controlled access methods.

7. Inheritance:
   ContactProcessModel and NeuralNetworkSurrogate inherit
   from SpreadModel.

8. Polymorphism:
   Different model objects provide their own versions of
   describe() and predict_growth().
"""
)