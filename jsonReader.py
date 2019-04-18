'''


with open('r.json') as json_file:
    data = json.load(json_file)
    for p in data['review']:
        print p['text']

'''
import json

with(open("review.json",'r')) as file_text:
    count_pos = 0
    count_neg = 0
    for i in file_text:
        dt = json.loads(i)
        if(count_pos<1000 or count_neg<1000):
    #        print count_neg
    #        print count_pos
            try:
                if dt['stars']>3.0:
                    file = open("yelp_txt_sentiment/pos/"+str(count_pos)+".txt", 'w')
                    file.write(dt['text'])
                    count_pos+=1
                    file.close()
                elif dt['stars']<3.0:
                    file = open("yelp_txt_sentiment/neg/" + str(count_neg) + ".txt", 'w')
                    file.write(dt['text'])
                    count_neg+=1
                    file.close()
            except:
                continue
#                print "Error"
        else:
            break
