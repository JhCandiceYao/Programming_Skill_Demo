"""The script is a module that contains three classes of apartments


Submitted by Candice Yao(jy3440)
This module implements different types of apartments, while the latter two inherit the methods of the first,
the super class. The subclasses add their type-specific methods on the top of the super class's methods.
"""


class Apartment:
    """a class that serve as the basic framework of latter subclasses. it includes the most basic info of an
    apartment """

    def __init__(self, id, rent):
        """The constructor takes two arguments: ID (an assigned string external to this script) and rent
        (monthly rent in dollars)The constructor specifies the following attributes:
        a) ID
        b) monthly rent
        c) the area in sq. ft. (assign 0 at construction; this is the total area of the apartment)"""
        self.ID = id
        self.monthly_rent = int(rent)
        self.area = 0
        self.type = "Apartment"

    def set_area(self, area):
        """A setter method for the area. It takes a single number corresponding to the area in
    square feet."""
        self.area = int(area)

    def get_area(self):
        """A getter method to report the area."""
        return self.area

    def price_per_unit_area(self):
        """A method to report the monthly price per unit of area. If the area is 0, raise a ValueError exception
        and display the message: “Area not defined."""
        if self.area == 0:
            raise ValueError("Area not defined.")
        return self.monthly_rent / int(self.area)

    def target_wages(self):
        """A general method for the target wages1 according to the following rule of thumb:
    monthly rent * 12 / 0.4"""
        return self.monthly_rent * 12 / 0.4

    def __str__(self):
        """a method to report information when print() and str() are called"""
        return f"Type: {self.type}, ID: {self.ID}, monthly rent: $ {self.monthly_rent}, area: {self.get_area()} sq.ft., " \
               f"price per unit of area: $ {self.price_per_unit_area()}, Wages required: ${self.target_wages()}"


class Studio(Apartment):
    """a subclass that inherits methods from apartment class, but it also implements specific attributes for studio"""

    def __init__(self, id, rent, has_separate_kitchen=True):
        """The constructor takes three arguments: ID (a number assigned somewhere else), rent
        (monthly rent in dollars) and whether it has a separate kitchen (as opposed to a kitchenette;
        defaults to True)."""
        super().__init__(id, rent)
        self.has_separate_kitchen = has_separate_kitchen
        self.type = "Studio"

    def set_has_separate_kitchen(self, has_separate_kitchen):
        """A setter method to mark it as having a separate kitchen space or not."""
        self.has_separate_kitchen = has_separate_kitchen

    def get_has_separate_kitchen(self):
        """A getter method to report whether it has a separate kitchen."""
        return self.has_separate_kitchen

    def get_type(self):
        """return the type of apartment"""
        return self.type

    def __str__(self):
        """a method to report information when print() and str() are called; it builds on the one provided by the
        Apartment class."""
        apartment_str = super().__str__()
        if self.has_separate_kitchen:
            return f"{apartment_str}, has a separate kitchen."
        else:
            return f"{apartment_str}, does not have a separate kitchen."


class TwoBedApartment(Apartment):
    """a subclass that inherits methods from apartment class, but it also implements specific attributes for two bed-room
    apartment"""

    def __init__(self, id, rent, has_roommate=False):
        """The constructor takes three arguments: ID (a number assigned somewhere else), rent
        (monthly rent in dollars) and whether it comes with a roommate (boolean, defaults to
        False).
        The constructor specifies the following attributes:
        a) ID
        b) monthly rent
        c) the area in sq. ft. (This behaves like the one in Apartment.)
        d) a dictionary with the areas for the “common areas,” “room 1” and “room 2.” (Initial
        values are 0.)
        e) whether it has someone living there already: a roommate. The roommate is assumed to
        be in the first room. There is no roommate by default."""
        super().__init__(id, rent)
        self.roommate = None
        self.type = "Two-bedroom apartment"
        self.room_areas = {"common areas": 0, "room 1": 0, "room 2": 0}
        self.roommate_already_there = has_roommate
        self.added_areas = 0

    def set_room_areas(self, areas: list):
        """A setter method to populate the dictionary with the areas of the different rooms; it takes a list of areas
        and assigns the first two numbers to the first room and second room respectively. The third area is assigned
        to the common room. A total area is calculated from the sum of the three rooms and named added areas. The common
        area has to be at least 20% of the total area, if not an error will be raised."""
        self.room_areas["room 1"] = areas[0]
        self.room_areas["room 2"] = areas[1]
        self.room_areas["common areas"] = areas[2]
        self.added_areas = sum(areas)
        if self.room_areas["common areas"] / self.added_areas < 0.2:
            raise ValueError("Common room is too small: " + str(self.room_areas))

    def set_area(self, area):
        """A method to set the area that overrides the inherited one. If the areas for the different parts of the
        apartment have been set, the common area will have to be modified to accommodate the new total area. If the
        common area comes out as less than 20% of the total area, it raises a ValueError and prints an appropriate
        message. If the areas have not been set, this method simply sets the total area."""
        if self.room_areas["common areas"] == 0 or self.room_areas["room 1"] == 0 or self.room_areas["room 2"] == 0:
            super().set_area(area)  # if different areas weren't set before, use super getter method
        else:
            self.room_areas["common areas"] = int(area) - (self.room_areas["room 1"] + self.room_areas["room 2"])
            print(f"updated common is {self.room_areas['common areas']}, int(area) is {int(area)}")
            self.area = int(self.room_areas["common areas"]) + int(self.room_areas["room 1"]) \
                        + int(self.room_areas["room 2"])
            # if different areas were already set: just set the total area to
            print(f"updated sum is {self.area} ")
            # the sum of the three rooms
            # check if the 20% rule is satisfied
            if (self.room_areas["common areas"] / int(self.area)) < 0.2:
                raise ValueError("Common room is too small: " + str(self.room_areas))

    def set_has_roommate(self, has_roommate):
        """A setter method to indicate whether it comes with a roommate."""
        self.roommate_already_there = has_roommate

    def get_has_roommate(self):
        """A getter method to report whether it comes with a roommate."""
        return self.roommate_already_there

    def set_roommate(self, given_name, family_name):
        """A setter method to add the roommate information: family name and given name. If
        there is no roommate before this assignment, it changes that attribute to True. We have to
        keep things consistent."""
        self.roommate = (given_name, family_name)
        if not self.roommate_already_there:
            self.roommate_already_there = True

    def price_per_unit_area(self):
        """A method to report the monthly price per unit of area. If the area is 0, it raises a
        ValueError exception with an appropriate message. If the apartment includes a roommate, the area to be used is
        half the area of the common areas and the area of the second room."""
        if self.area == 0:
            raise ValueError("Area not defined.")
        if self.roommate_already_there:
            return self.monthly_rent / (self.room_areas["common areas"] / 2 + self.room_areas["room 2"])
        else:
            return self.monthly_rent / self.area

    def __str__(self):
        """A method to report the information built upon the one provided by the Apartment class."""
        apartment_str = super().__str__()
        if self.roommate:
            return f"{apartment_str}, has a roommate named{self.roommate[0]} {self.roommate[1]}"
        else:
            return apartment_str

    def get_type(self):
        """A method to report the type of apartment."""
        return self.type
