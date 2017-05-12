import sys
def progress(i, end_val,bar_length=50,addition=""): 
    '''
    Print a progress bar of the form: Percent: [#####      ]
    i is the current progress value expected in a range [0..end_val]
    bar_length is the width of the progress bar on the screen.
    use in an n long loop: progress(i,n,50)
    '''
    percent = float(i) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [{0}] {1}% {2}".format(hashes + spaces, int(round(percent * 100)),addition))
    sys.stdout.flush()
