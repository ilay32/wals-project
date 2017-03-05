import re,logging

### globals ####
"""
match first numeric sequence as group(1), and make sure there are no
other numbers after it.
"""
numerizer = re.compile("(^\d+)([^\d]*$)")
"""
(tmp?) fallback for failed numerization.
"""
simplenumerizer = re.compile("(^\d+)")

### extractors ###
def natural(c):
    """
    just get the numeric value of the cell
    """
    return numerize(c)

def mult2bin(target_value,value):
    """
    binarize a multi-valued feature, returning -1 if the value is n, 
    and 1 otherwise, returns the function that does that
    """
    def which(c):
        return value if numerize(c) == target_value else -1*value
    return which


### helpers ###
def numerize(txt):
    """
    if there's no match, it means there is more
    than one numeric sequence in the cell, in which
    case, print the cell contents so, we can see what's what
    """
    m = numerizer.match(txt)
    if m:
        return int(m.group(1))
    else:
        logging.warning("can't numerize cell contents: %s",txt)
        return int(simplenumerizer.match(txt).group(1))

