import csv
import datetime
import json
import re


regExPat = r'"note_desc" : (\w*),'
regExRep = r'"note_desc" : "\1",'

regExPat2 = r',([^,]*})'
regExRep2 = r'\1'


def readTimeStampData(filePath, dateBegin=None, dateEnd=None):
    """
        Read either .csv file or .json file and converts all timestamps to datetime objects within
        a given time interval specified by DateBegin and DateEnd. Function
        defaults to reading all the file if no time interval is specified.

        Inputs:
            filePath: A string that specifies the file path
            to timestamper file (either .csv or .json file)

            dateBegin: A string in the format: "YYYY-MM-DD" or
            "YYYY-MM-DD HH:MM"
            Example: DateBegin="2021-10-12 14:32"

            dateEnd: A string in the format: "YYYY-MM-DD" or
            "YYYY-MM-DD HH:MM"

        Output:
            A list of tuples where each tuple represents a time stamp.
            Tuple has the following format (DateTime object, String Note)
    """
    # construct datetime objects from the given time interval
    if (dateBegin is None):
        beginDateTime = datetime.datetime(1999, 1, 1)
    elif (len(dateBegin) > 10):
        beginDateTime = datetime.datetime.strptime(dateBegin, '%Y-%m-%d %H:%M')
    else:
        beginDateTime = datetime.datetime.strptime(dateBegin, '%Y-%m-%d')
    #print(beginDateTime)

    if (dateEnd is None):
        endDateTime = datetime.datetime(2050, 1, 1)
    elif (len(dateEnd) > 10):
        endDateTime = datetime.datetime.strptime(dateEnd, '%Y-%m-%d %H:%M')
    else:
        endDateTime = datetime.datetime.strptime(dateEnd, '%Y-%m-%d')
    #print(endDateTime)

    # initialize output variable
    timeStamps = []


    # open the file
    with open(filePath) as fH:

        # check if the file is csv file
        if filePath.lower().endswith('.csv'):

            csv_reader = csv.reader(fH)

            # iterate each row of the .csv file
            for ind, row in enumerate(csv_reader):
                if ind >= 9: # disregard the first 9 rows
                    # get the string data
                    dateStr = row[3]
                    timeStr = row[4]
                    noteStr = row[1]

                    # convert strings to datetime object
                    timeStamp = datetime.datetime.strptime(dateStr + ' ' + timeStr, "%d-%b-%Y %I:%M:%S.%f %p")

                    # check if the timestamp is within the time interval
                    if (timeStamp > beginDateTime and timeStamp < endDateTime):

                        # print(timeStamp)
                        # print(noteStr)
                        # If so, then save the timestamp.
                        timeStamps.append((timeStamp, noteStr))

        # check if the file is .json file
#         elif (filePath.lower().endswith('.json')):
        elif (filePath.lower().endswith('.json')):

            with open(filePath, 'r') as f:
                json_content = json.load(f)['note']

            timeStamps = []

            for row in json_content:
                timeStamp = datetime.datetime.strptime(row['dateNtime'], "%Y%m%d%H%M%S%f")
                noteStr = row['note_desc']
                if (timeStamp > beginDateTime and timeStamp < endDateTime):
                    timeStamps.append((timeStamp, noteStr))


        elif (filePath.lower().endswith('.txt')):
            fileContent = fH.read()

            # need to modify incorrectly formatted .json file

            # remove everything before the note object
            fileContent = fileContent[fileContent.find('note ='):]
            fileContent = '{ ' + fileContent

            # replace keys with proper strings
            wordsToReplace = ['note', 'date', 'dateNtime', 'latitude', 
                'longitude', 'time', 'user']
            for aWord in wordsToReplace:
                fileContent = fileContent.replace(aWord + ' ', '"' + aWord + '" ')
            
            # replace all incorrect punctuation
            fileContent = fileContent.replace(';', ',')
            fileContent = fileContent.replace('(', '[')
            fileContent = fileContent.replace(')', ']')
            fileContent = fileContent.replace('=', ':')

            

            # replace some notes where the value is not a string
            fileContent = re.sub(regExPat, regExRep, fileContent)


            # delete the comma at the end of an array.
            fileContent = re.sub(regExPat2, regExRep2, fileContent)

            
#             print(fileContent)
#             print(fileContent)

            jsonData = json.loads(fileContent)

            # iterate through the timestamps
            for aNote in jsonData['note']:
                # convert strings to datetime object
                dateStr = aNote['date'] + ' ' + aNote['time']
                timeStamp = datetime.datetime.strptime(dateStr, "%d-%m-%Y %H:%M:%S:%f")

                    # check if the timestamp is within the time interval
                if (timeStamp > beginDateTime and timeStamp < endDateTime):

#                     print(timeStamp)
#                     print(aNote['note_desc'])
                        # If so, then save the timestamp.
                    timeStamps.append((timeStamp, aNote['note_desc']))            
            

        else:
            raise ValueError('File must be either a .csv or .txt or .json file')

                
    return timeStamps


if __name__ == "__main__":
    csvFilePath = 'tm_backup_10_12_21_102657.csv'
    jsonFilePath = 'tm_backup_1210.json'
    readTimeStampData(jsonFilePath, '2021-12-09', '2021-12-11')