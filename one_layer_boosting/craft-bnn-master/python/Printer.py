
def print_3d(img, N, rows, cols, fmt='f'):
    for n in range(N):
        for r in range(rows):
            for c in range(cols):
                if fmt=='b':
                    print "%2u " % (0 if img[n,r,c] >= 0 else -1),
                elif fmt=='i':
                    print "%5d " % img[n,r,c],
                else:
                    print "%6.3f " % img[n,r,c],
            print ""
        print " ##"

def print_2d(img, rows, cols, fmt='f'):
    for r in range(rows):
        for c in range(cols):
            if fmt=='b':
                print "%2u " % (0 if img[r,c] >= 0 else -1),
            elif fmt=='i':
                print "%5d " % img[r,c],
            else:
                print "%6.3f " % img[r,c],
        print ""


