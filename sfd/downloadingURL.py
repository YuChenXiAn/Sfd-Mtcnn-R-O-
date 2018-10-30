import openpyxl
import urllib.request
import os
import logging

book = openpyxl.load_workbook('/home/xdepartment/ly/docs/registered.xlsx')
sheet = book.active
#pingList = sheet['A2':'A90049']
urlList = sheet['B2':'B90049']

savePath = '/home/xdepartment/ly/data/jdface/registered90k'
logging.basicConfig(filename='downloadingError.log',level=logging.DEBUG)
count = 1
for url, in urlList:
    print(count)
    count = count + 1
    
    urlPath = url.value
    file_name,file_extension = os.path.splitext(urlPath)
    if file_extension == '.jpg':
       file_name = savePath + '/' + str(count) + '.jpg'
    else:
       file_name = savePath + '/' + str(count) + '.png'
    try:
       urllib.request.urlretrieve(urlPath, file_name)
    except:
       logging.debug('Error downloading:'+str(count))

    

