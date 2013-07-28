#!/usr/local/bin/python
from mrjob.job import MRJob
import itertools

# ex1 - word count
class MRWC(MRJob):

    def mapper(self, _, line):

        # print mapper input
        # print line

        for word in line.split():
            # print mapper output
            # print word, 1
	# instead of using the word we are using a static keyword
	# mapper is quivalent to GROUP BY of SQL, before we were doing something	# like SELECT count(word) GROUP BY word
	# now we got rid of the GROUP BY and doing something like 
	# SELECT COUNT(word)
	
            yield "totalCt", 1

    def reducer(self, word, counts):

        # print reducer input
        # note - counts is a generator object, so we need to do some fancy
        #        stuff to copy it (otherwise it's only good for a single use)
        # tmp1, tmp2 = itertools.tee(counts)
        # print word, [k for k in tmp1]
        # yield word, sum(tmp2)

        yield "totalCt", sum(counts)

if __name__ == '__main__':
    MRWC.run()
    # MRInvIdx.run()
    # MRWC2.run()
