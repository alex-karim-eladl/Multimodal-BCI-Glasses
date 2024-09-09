"""
File:			/run.py
Project:		experiment
Author:		Alex Karim El Adl

Description:
"""

import argparse
import json
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import sys
import time
from tkinter import *

import chime

# from blueberry.record import
# from attentivu.streamer import run_streamer
from gtts import gTTS
from playsound import playsound

from protocol import *


# generate timings for random timing tasks
def generate_start_times():
    print('generating start times...')
    math_start = [0, 0, 0]
    rest_start = [0, 0, 0]

    # for all three random parts
    for i in range(3):
        start_times = []
        while len(start_times) < 2 * ROUNDS:
            new_time = random.randint(1, RANDOM_PART_TIME - PERIOD_TIME - 1)
            if all(abs(new_time - t) > PERIOD_TIME for t in start_times):
                start_times.append(new_time)

        # random.shuffle(start_times)
        start_times = sorted(start_times)

        math_start[i] = start_times[::2]
        rest_start[i] = start_times[1::2]

    return math_start, rest_start


# generate problems
def generate_problems(stimulus, timing):
    print(f'generating {stimulus} problems...')
    problems = [[''] * VISUAL_PROBLEM_NUM for i in range(ROUNDS)]
    for j in range(ROUNDS):
        for k in range(VISUAL_PROBLEM_NUM):
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            if b < a:
                problems[j][k] = f'. {b}\n+ {a}'
            else:
                problems[j][k] = f'. {a}\n+ {b}'

            if stimulus == 'auditory':
                audio = gTTS(text=problems[j][k], lang='en', slow=False)
                audio.save('audio_files/' + timing + '/' + str(j) + '_' + str(k) + '.mp3')
    return problems


def key_press(event):
    global waiting, current_instruction
    if event.char == ' ' and waiting:
        waiting = False

        # part 1
        if current_instruction == 3:
            visual_practice()
            # pass
        elif current_instruction == 4:
            visual_regular()
            # auditory_regular()
            # mental_regular()

        # part 2:
        elif current_instruction == 5:
            auditory_practice()
            # pass
        elif current_instruction == 6:
            # visual_regular()
            auditory_regular()
            # mental_regular()

        # part 3:
        elif current_instruction == 7:
            mental_practice()
            # pass
        elif current_instruction == 8:
            # visual_regular()
            # auditory_regular()
            mental_regular()

        # part 4:
        elif current_instruction == 10:
            visual_random()
            # auditory_random()
            # mental_random()
        # part 5:
        elif current_instruction == 11:
            # visual_random()
            auditory_random()
            # mental_random()
        # part 6:
        elif current_instruction == 12:
            # visual_random()
            # auditory_random()
            mental_random()
            finish()

        if current_instruction != len(INSTRUCTIONS):
            display_text(INSTRUCTIONS[current_instruction][0], INSTRUCTIONS[current_instruction][1])
            current_instruction += 1
            waiting = True


def start():
    global waiting, current_instruction
    display_text(
        'Thank you for agreeing to participate in this experiment!\nPlease press space to begin the tutorial.', 50
    )

    # restart at part if specified in command line args
    if args.part == 2:
        current_instruction = 5
    elif args.part == 3:
        current_instruction = 7
    elif args.part == 4:
        current_instruction = 10
    elif args.part == 5:
        current_instruction = 11
    elif args.part == 6:
        current_instruction = 12

    waiting = True


def finish():
    # pprint.pprint(timing)
    display_text('You are done\nThank you for participating in our experiment!', 50)
    time.sleep(5)

    p1.kill()
    p2.kill()
    p3.kill()
    p4.kill()
    win.destroy()  ###########
    sys.exit(0)


def handler(signum, frame):
    raise Exception('')


# break and rest periods
def non_math(duration: float, text: str, countdown: bool):
    if duration != 0:
        start_time = time.time()
        if countdown:  # display time remaining every second
            for s in range(duration, 0, -1):
                display_text(str(s) + 's ' + text)
                time.sleep(1 - ((time.time() - start_time) % 1))
        else:  # does not display remaining time (random parts)
            display_text(text)
            time.sleep(float(duration) - ((time.time() - start_time) % float(duration)))


def display_text(text, size=100):
    label.config(text=text, font=('none bold', size))
    win.update()


# practice_problems = generate_practice_problems()


def visual_practice():
    start_time = time.time()
    chime.success()

    # MATH
    for i in range(VISUAL_PROBLEM_NUM):
        display_text(practice_problems[i])
        time.sleep(VISUAL_PROBLEM_TIME - ((time.time() - start_time) % VISUAL_PROBLEM_TIME))

    chime.success()
    # REST
    non_math(PERIOD_TIME, 'Rest', True)


def auditory_practice():
    start_time = time.time()
    chime.success()

    # MATH
    display_text('Auditory Math')
    for i in range(AUDITORY_PROBLEM_NUM):  # reps = how many problems per math block
        playsound('./audio_files/practice/' + str(i) + '.mp3')
        time.sleep(AUDITORY_PROBLEM_TIME - ((time.time() - start_time) % AUDITORY_PROBLEM_TIME))


def mental_practice():
    start_time = time.time()
    chime.success()

    # MATH
    display_text('Mental Math')
    time.sleep(PERIOD_TIME - ((time.time() - start_time) % PERIOD_TIME))


# part 1
def visual_regular():
    global timing

    non_math(10, 'Rest', True)
    start_time = time.time()
    print('\nvisual regular', start_time)

    for j in range(ROUNDS):
        chime.success()
        # MATH
        for k in range(VISUAL_PROBLEM_NUM):
            display_text(visual_problems_regular[j][k])
            time.sleep(VISUAL_PROBLEM_TIME - ((time.time() - start_time) % VISUAL_PROBLEM_TIME))

        chime.success()
        # REST
        non_math(PERIOD_TIME, 'Rest', True)

    end_time = time.time()
    print(end_time - start_time)

    timing['visual_regular_start'] = start_time
    timing['visual_regular_end'] = end_time
    with open(f'./data/{args.subject}/blueberry/{args.runname}timing.pkl', 'wb') as f:
        pickle.dump(timing, f)

    # BREAK
    non_math(BREAK_TIME, 'Break', True)


# part 2
def auditory_regular():
    global timing

    non_math(10, 'Rest', True)
    start_time = time.time()
    print('\nauditory regular', start_time)

    for j in range(ROUNDS):
        chime.success()
        # MATH
        display_text('Auditory Math')
        for k in range(AUDITORY_PROBLEM_NUM):  # reps = how many problems per math block
            playsound('audio_files/regular/' + str(j) + '_' + str(k) + '.mp3')
            time.sleep(AUDITORY_PROBLEM_TIME - ((time.time() - start_time) % AUDITORY_PROBLEM_TIME))

        chime.success()
        # REST
        non_math(PERIOD_TIME, 'Rest', True)

    end_time = time.time()
    print(end_time - start_time)

    timing['auditory_regular_start'] = start_time
    timing['auditory_regular_end'] = end_time
    with open(f'./data/{args.subject}/blueberry/{args.runname}timing.pkl', 'wb') as f:
        pickle.dump(timing, f)

    non_math(BREAK_TIME, 'Break', True)


# part 3
def mental_regular():
    global timing

    non_math(10, 'Rest', True)
    start_time = time.time()
    print('\nmental regular', start_time)

    for j in range(ROUNDS):
        chime.success()
        # MATH
        display_text('Mental Math')
        time.sleep(PERIOD_TIME - ((time.time() - start_time) % PERIOD_TIME))

        chime.success()
        # REST
        non_math(PERIOD_TIME, 'Rest', True)

    end_time = time.time()
    print(end_time - start_time)

    timing['mental_regular_start'] = start_time
    timing['mental_regular_end'] = end_time
    with open(f'./data/{args.subject}/blueberry/{args.runname}timing.pkl', 'wb') as f:
        pickle.dump(timing, f)

    non_math(BREAK_TIME, 'Break', True)


# part 4
def visual_random():
    global timing
    rest = PERIOD_TIME
    math_start = math_start_times[0]
    math_start.append(RANDOM_PART_TIME)  # to calc last rest

    non_math(10, 'Rest', False)
    start_time = time.time()
    print('\nvisual random', start_time)
    print(math_start)

    chime.success()
    # REST
    non_math(math_start[0], 'Rest', False)
    for j in range(ROUNDS):
        chime.success()
        last = time.time()
        # MATH
        for k in range(VISUAL_PROBLEM_NUM):  # reps = how many problems per math block
            display_text(visual_problems_random[j][k])
            time.sleep(VISUAL_PROBLEM_TIME - ((time.time() - last) % VISUAL_PROBLEM_TIME))

        # REST
        rest = math_start[j + 1] - (math_start[j] + PERIOD_TIME)
        non_math(rest, 'Rest', False)

    end_time = time.time()
    print(end_time - start_time)

    timing['visual_random_start'] = start_time
    timing['visual_random_end'] = end_time
    with open(f'./data/{args.subject}/blueberry/{args.runname}timing.pkl', 'wb') as f:
        pickle.dump(timing, f)

    non_math(BREAK_TIME, 'Break', True)


# part 5
def auditory_random():
    global timing
    rest = PERIOD_TIME
    math_start = math_start_times[1]
    math_start.append(RANDOM_PART_TIME)  # to calc last rest

    non_math(10, 'Rest', False)
    start_time = time.time()
    print('\nauditory random', start_time)
    print(math_start)

    # REST
    non_math(math_start[0], 'Rest', False)
    for j in range(ROUNDS):
        chime.success()
        last = time.time()
        # MATH
        display_text('Auditory Math')
        for k in range(AUDITORY_PROBLEM_NUM):
            playsound('audio_files/random/' + str(j) + '_' + str(k) + '.mp3')
            time.sleep(AUDITORY_PROBLEM_TIME - ((time.time() - last) % AUDITORY_PROBLEM_TIME))

        chime.success()
        # REST
        rest = math_start[j + 1] - (math_start[j] + PERIOD_TIME)
        non_math(rest, 'Rest', False)

    end_time = time.time()
    print(end_time - start_time)

    timing['auditory_random_start'] = start_time
    timing['auditory_random_end'] = end_time
    with open(f'./data/{args.subject}/blueberry/{args.runname}timing.pkl', 'wb') as f:
        pickle.dump(timing, f)

    non_math(BREAK_TIME, 'Break', True)


# part 6
def mental_random():
    global timing
    rest = PERIOD_TIME
    math_start = math_start_times[2]
    math_start.append(RANDOM_PART_TIME)  # to calc last rest

    non_math(10, 'Rest', False)
    start_time = time.time()
    print('\nmental random', start_time)
    print(math_start)

    # REST
    non_math(math_start[0], 'Rest', False)
    for j in range(ROUNDS):
        chime.success()
        last = time.time()
        # MATH
        display_text('Mental Math')
        time.sleep(PERIOD_TIME - ((time.time() - last) % PERIOD_TIME))

        chime.success()
        # REST
        rest = math_start[j + 1] - (math_start[j] + PERIOD_TIME)
        non_math(rest, 'Rest', False)

    end_time = time.time()
    print(end_time - start_time)

    timing['mental_random_start'] = start_time
    timing['mental_random_end'] = end_time
    with open(f'./data/{args.subject}/blueberry/{args.runname}timing.pkl', 'wb') as f:
        pickle.dump(timing, f)

    non_math(BREAK_TIME, 'Break', True)


if __name__ == 'main':
    # parse arguments
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('subject', type=str, help='subject identifier')
    parser.add_argument('runname', type=str, help='run indentifier')
    args = parser.parse_args()

    # create directories
    os.makedirs('./audio_files/regular', exist_ok=True)
    os.makedirs('./audio_files/random', exist_ok=True)
    os.makedirs('./audio_files/practice', exist_ok=True)
    os.makedirs(f'./data/{args.subject}/blueberry', exist_ok=True)

    math_start_times, rest_start_times = generate_start_times()
    visual_problems_regular = generate_problems('visual', 'regular')
    visual_problems_random = generate_problems('visual', 'random')
    generate_problems('auditory', 'regular')
    generate_problems('auditory', 'random')

    # def generate_practice_problems():
    practice_problems = [''] * VISUAL_PROBLEM_NUM
    for k in range(VISUAL_PROBLEM_NUM):
        a1 = a2 = b1 = b2 = 0
        while a1 + b1 < 10 and a2 + b2 < 10:
            a1 = random.randint(0, 9)
            a2 = random.randint(0, 9)
            b1 = random.randint(0, 9)
            b2 = random.randint(0, 9)
        a = a1 * 10 + a2
        b = b1 * 10 + b2
        practice_problems[k] = str(a) + '\n+ ' + str(b)
        if b > a:
            practice_problems[k] = str(b) + '\n+ ' + str(a)
        else:
            practice_problems[k] = str(a) + '\n+ ' + str(b)

        # save audio
        audio = gTTS(text=practice_problems[k], lang='en', slow=False)
        audio.save('./audio_files/practice/' + str(k) + '.mp3')

    waiting = False
    current_instruction = 0

    # setup tkinter
    win = Tk()
    win.configure(background='black')
    win.attributes('-fullscreen', True)
    label = Label(win, text='', font='none 100 bold', bg='black', fg='white')
    # label.pack(padx=50,pady=250)
    label.place(relx=0.5, rely=0.5, anchor=CENTER)

    win.bind('<KeyPress>', key_press)
    win.update()

    signal.signal(signal.SIGALRM, handler)

    # save timing info to file
    timing = {
        'visual_random': {math_start_times[0], rest_start_times[0]},
        'auditory_random': {math_start_times[1], rest_start_times[1]},
        'imagined_random': {math_start_times[2], rest_start_times[2]},
    }

    with open(f'../data/{args.subject}/timing.json', 'a+') as f:
        json.dump(timing, f)

    with Path(__file__).parent.joinpath('streaming', 'blueberry', 'devices.json').open() as f:
        devices = json.load(f)

    # FIXME: refactor - hack needed for timing precision?
    bby_path = Path(__file__).parent.joinpath('streaming', 'blueberry', 'record.py')
    att_path = Path(__file__).parent.joinpath('streaming', 'attentivu', 'streamer.py')
    # start recieving data
    try:
        p1 = subprocess.Popen(f"python3 {bby_path} -a {devices['f7']}  -u {args.subject} -r {args.runname}", shell=True)
        p2 = subprocess.Popen(f"python3 {bby_path} -a {devices['f8']} -u {args.subject} -r {args.runname}", shell=True)
        p3 = subprocess.Popen(f"python3 {bby_path} -a {devices['fp']} -u {args.subject} -r {args.runname}", shell=True)
        p4 = subprocess.Popen(f'python3 {att_path} {args.subject} {args.runname}', shell=True)
        time.sleep(5)
    except KeyboardInterrupt:
        raise Exception('User stopped program.')

    win.after(0, start())
    win.mainloop()
