"""Takes a text string and modifies it according to some odd rules


Submitted by Mauricio Arias. NetID: ma6918
This script takes a string and substitutes cryptic summaries for words
longer than 2 characters. It uses multiple assignments.
"""

def summarize_text(input_string):
    # Separate the elements using spaces as the marker.
    elements = [elem.strip() for elem in input_string.split(" ")]
    summarized_elements = []
    for elem in elements:
        # Each element is considered composed of pre-symbols, the actual
        # word and post-symbols. Hence, words in quotes will be identified
        # properly. However, some cases are identified improperly:
        # for example "The dogs' tails moved at the same time." The
        # script considers the word "dogs'" to be the word dogs followed by
        # the punctuation mark "'".
        # Notice that "pre-made" is considered a single word: as intended.
        beginning = -1
        end = len(elem) - 1
        # Symbols at the beginning...
        for pos, char in enumerate(elem):
            if char.isalpha():
                beginning = pos
                break
        # Symbols at the end. If there are only symbols in the element,
        # there is no need to check anything else.
        if not beginning == -1:
            for pos, char in enumerate(elem[::-1]):
                if char.isalpha():
                    end = len(elem) - pos
                    break
            # The middle word is what is left in the middle
            middle_word = elem[beginning: end]
            if len(middle_word) > 2:
                first, *middle, last = middle_word
                summarized_word = first + str(len(middle)) + last
            else:
                summarized_word = middle_word
        else:
            summarized_word = ""

        summarized_elements.append(elem[:beginning] + summarized_word + elem[end:])

    return " ".join(summarized_elements)

user_string = "This is a simple, but useful example! However, there are many cases not included here..."
# user_string = "This is pre-made lemonade. However, it tastes like it was 'homemade' a week ago."
user_string = """Testing the behavior of empty words ;: "; ""."""
# user_string = input("Provide an input string: ")
# print(summarize_text(user_string))

