# experiment timing definitions in seconds
VISUAL_PROBLEM_TIME = 4  # seconds
VISUAL_PROBLEM_NUM = 7  # number of problems in a visual math period
AUDITORY_PROBLEM_TIME = 7  # seconds
AUDITORY_PROBLEM_NUM = 4  # number of problems in an auditory math period

PERIOD_TIME = VISUAL_PROBLEM_NUM * VISUAL_PROBLEM_TIME  # math and rest period length
BREAK_TIME = 120  # between parts
ROUNDS = 10  # sets of math and rest in a part
REGULAR_PART_TIME = 2 * PERIOD_TIME * ROUNDS
RANDOM_PART_TIME = 3 * PERIOD_TIME * ROUNDS


INSTRUCTIONS = [
    # [f'''Mental Math\nNo Stimuli''', 100],
    [
        f"""The experiment has 6 parts and instructions for each part will be displayed before it starts.
Parts 1-3 are {round(REGULAR_PART_TIME/60, 1)} minutes each and parts 4-6 are {round(RANDOM_PART_TIME/60, 1)} minutes each.
You will be given {BREAK_TIME/60} minute breaks between the parts in which no data will be collected.
You can use these breaks to ask the experimenter any questions you have.
Please only adjust the glasses during the breaks and limit movements outside of the breaks.\n
Press space to continue.""",
        25,
    ],
    [
        f"""Part 1: Regular Timing with Visual Stimuli\n
Part 1 consists of alternating {PERIOD_TIME} second math periods and {PERIOD_TIME} second rest periods.
During a math period you will see a new addition problem displayed every {VISUAL_PROBLEM_TIME} seconds.
Please do your best to solve each problem in your head but move on to the next one once it appears.
It's okay if you run out of time or get some problems wrong
as long as you remain focused and do your best to solve them.\n
Press space to continue.""",
        25,
    ],
    [
        f"""Part 1: Regular Timing with Visual Stimuli\n
After each math period a {PERIOD_TIME} second rest period
will begin during which you should do nothing.\n
Press space to practice one math and one rest period.""",
        30,
    ],
    # visual_practice() 3
    [
        f"""Part 1: Regular Timing with Visual Stimuli\n
Great! There will be {ROUNDS} such sets of math and rest periods in part 1.\n
Press space to begin part 1.""",
        30,
    ],
    # visual_regular() 4
    [
        f"""Part 2: Regular Timing with Auditory Stimuli\n
Part 2 consists of alternating {PERIOD_TIME} second math periods
and {PERIOD_TIME} second rest periods just like part 1.
However, the problems will be read to you rather than displayed on the screen.
You will hear a new problem every {AUDITORY_PROBLEM_TIME} seconds.
Please try your best to solve them just like before.\n
Press space to practice an auditory math period.""",
        30,
    ],
    # auditory_practice() 5
    [
        f"""Part 2: Regular Timing with Auditory Stimuli\n
Great! There will again be {ROUNDS} sets of math and rest periods in part 2.\n
Press space to begin part 2.""",
        30,
    ],
    # auditory_regular() 6
    [
        """Part 3: Regular Timing with No Stimuli\n
Part 3 is the same as parts 1-2 except there will be no stimuli during math periods.
Instead the screen will say 'Mental Math'.
During this time you should come up with addition problems with similar length and
difficulty to those in parts 1-2 in your head and try to solve them like before.\n
Press space to practice one such mental math period.""",
        30,
    ],
    # mental_practice() 7
    [
        f"""Part 3: Regular Timing with No Stimuli\n
Great! There will again be {ROUNDS} sets of math and rest periods in part 3.\n
Press space to begin Part 3.""",
        30,
    ],
    # mental_regular() 8
    [
        f"""Parts 4-6 are each {round(RANDOM_PART_TIME/60,1)} minutes long.
Ten {PERIOD_TIME} second math periods will begin at random times.
During the remainder of the {round(RANDOM_PART_TIME/60,1)} minutes
the screen will say 'Rest' and you should again do nothing.\n
Press space to continue.""",
        30,
    ],
    [
        f"""Part 4: Random Timing with Visual Stimuli\n
The math periods in part 4 will have visual stimuli like in part 1.
Please solve the problems in the same way as before once they appear.\n
There is no practice period for part 4. Press space to begin part 4.""",
        30,
    ],
    # visual_random() 10
    [
        f"""Part 5: Random Timing with Auditory Stimuli\n
Part 5 is the same as part 4 except the math periods have auditory stimuli like part 2.\n
There is no practice period for part 5. Press space to begin part 5.""",
        30,
    ],
    # auditory_random() 11
    [
        f"""Part 6: Random Timing with No Stimuli\n
Part 6 is the same as parts 4-5 except the math periods have no stimuli like part 3.
Please come up with and solve problems in the same way as
part 3 when the screen says 'Mental Math'.\n
There is no practice period for part 6. Press space to begin part 6.""",
        30,
    ],
    # mental_random() 12
]
