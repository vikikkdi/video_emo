import csv
import emotion_detection
import sys

if __name__=='__main__':
	scenes = []
	with open('sceneout.csv','r') as f:
		reader = csv.reader(f)
		for i in reader:
			scenes.append(i)

	imp_data = [(i[0],i[1]) for i in scenes[2:]]

	processData = []
	frameStart = 0

	for i in imp_data:
		processData.append((int(i[0]),frameStart,int(i[1])))
		frameStart = int(i[1])+1

	print("Data process completed")

	emotionData = []

	count = 0

	for i in processData:
		print(sys.argv[1:],i[1],i[2])
		emotionData.append([int(i[0]),emotion_detection.process(sys.argv[1:],i[1],i[2]), int(i[1]), int(i[2])])
		if count == 1:
			break
		count += 1

	print(emotionData)		

	ofile = open('output.csv','w')
	writer = csv.writer(ofile)

	for i in emotionData:
		writer.writerow(i)
	ofile.close()