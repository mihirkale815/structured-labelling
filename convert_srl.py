import pickle

def convert_format(data,outpath):
	f = open(outpath,"w")
	for d in data:
		sentence,_,tags = d
		for token,tag in zip(sentence,tags):
			f.write(tag + "\t" + token.text + "\n")
		f.write("\n")
	f.close()

dataset = pickle.load(open("srl_data.p","rb"))
l = len(dataset)
num_train = int(l*0.8)
num_dev = int(l*0.1)
num_test = int(l*0.1)

traindata = dataset[0:num_train]
devdata = dataset[num_train:num_train+num_dev]
testdata = dataset[num_train+num_dev:]

convert_format(traindata,"train.iob")
convert_format(devdata,"dev.iob")
convert_format(testdata,"test.iob")
