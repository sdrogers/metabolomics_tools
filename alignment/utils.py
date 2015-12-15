import sys

def draw_progress_bar(percent, barLen = 20):
    '''Draws progress bar, from http://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage'''
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()