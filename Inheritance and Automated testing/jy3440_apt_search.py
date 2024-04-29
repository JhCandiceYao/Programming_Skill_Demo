"""The script is a main script that uses the classes imported from the the module I created. It implements the methods
and request input from users to display results.

Submitted by Candice Yao(jy3440)
This module first requests users to input the type of apartment they are logging. With if statements, different types of
objects will be created accordingly. Once all the information of an object has been processed, the object will be store
a list called apartments - it will be printed at the end of the user's search if they press enter to exit the search.

It is assumed in the handling of the two-bedroom apartment that the user enters the total area first and then enter
individual room areas. The first enters total area is set to be correct, and if the value between the total area and the
added individual room areas do not match, the common room's area will be adjusted to meet the entered "total area".
"""
from jy3440_apartments import Apartment, Studio, TwoBedApartment

# Create a list of apartments
apartments = []
history_searches = []

while True:
    # Request type of apartment
    request_input = input("Please select the apartment type from the options below."
                          "\nPress Enter to finish apartment input. Apartment types available(1/2): \n1. Studio\n2. "
                          "Two bedroom apartment\n\n\nApartment type: ")
    if not request_input:
        for num, apartment in enumerate(apartments):
            print(f"You've looked up the following apartments.\n{num+1}. {apartment}\n")

        break
    # Request information shared by each type of apartment
    ID = input("Apartment ID: ")
    rent = int(input("Monthly rent: $"))
    area = input("Apartment area in square feet: ")
    print(f"now the area is still {area}")
    if request_input.lower() == "1":
        # Create studio apartment
        kitchen = input("Does the apartment have a separate kitchen? (Y/N): ")
        apartment = Studio(ID, rent, kitchen)
        apartment.area = area
        apartment.has_separate_kitchen = True if kitchen == "Y" else False

    elif request_input.lower() == "2":
        # Request information about two-bedroom apartment
        roommate = input("Does the apartment come with a roommate? (Y/N) ")
        # Create two-bedroom apartment
        apartment = TwoBedApartment(ID, rent, roommate)
        apartment.area = area
        print(f"now the area is still {area}, apartment.area is {apartment.area}")

        if roommate == "y" or "Y":
            apartment.set_has_roommate(True)
            # Request information about rooms in two-bedroom apartment
        room_areas = []
        for room in ["first", "second", "common"]:
            room_area = int(input(f"Provide the area of the {room} room in square feet: "))
            room_areas.append(room_area)
        apartment.set_room_areas(room_areas)
        print(f"now the added sum is {apartment.added_areas} while the area attribute is {apartment.area}")

        if apartment.added_areas != apartment.area:  # if the sum of the room areas is not the same as the total area
            # inputted by the user, we set the total according to user's input, and alter the common area based on it
            print(f"the two results are different, added is {apartment.added_areas} and reported is {apartment.area}")
            apartment.set_area(apartment.area)
        for area in apartment.room_areas.values():
            print(area)
        if apartment.roommate_already_there:
            # Request information about roommate
            given_name = input("Enter given name of roommate: ")
            family_name = input("Enter family name of roommate: ")
            apartment.set_roommate(given_name, family_name)

    print(apartment)
    print()
    apartments.append(apartment)
