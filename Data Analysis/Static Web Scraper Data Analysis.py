# import libraries
import csv

from bs4 import BeautifulSoup
import os
import requests
import time
import datetime
import smtplib
import csv
import pandas as pd

# Connect to Website and pull in data

def check_price():
    URL = 'https://www.amazon.com/Funny-Data-Systems-Business-Analyst/dp/B07FNW9FGJ/ref=sr_1_3?dchild=1&keywords=data%2Banalyst%2Btshirt&qid=1626655184&sr=8-3&customId=B0752XJYNL&th=1'
# Find Your User-Agent: https://httpbin.org/get
# Use user agent link to populate headers variable


“””

The user agent is a string of text that identifies the web browser, operating system, and device used to make a request to a web server. It helps websites understand how to render content properly for different devices and browsers. 

Identifying the user agent in your web scraping requests can be important. It helps mimic different web browsers or devices, making your requests appear more legitimate and reducing the chances of being blocked by websites.

“””

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
    "X-Amzn-Trace-Id": "Root=1-61634104-58c124fe1a51a9cb72bb420a",
    "X-Client-Data": "CIu2yQEIo7bJAQjEtskBCKmdygEI/4XLAQjq8ssBCO/yywEInvnLAQjegcwBCN2EzAEI54TMAQi2hcwBCNaFzAEI/4XMAQiBhswBGIyeywEYhv3LAQ=="}

    page = requests.get(URL, headers=headers)

    soup1 = BeautifulSoup(page.content, "html.parser")
    dataset = BeautifulSoup(soup1.prettify(), "html.parser")
    product_information = dataset.find(id="productTitle").get_text()
    product_price = dataset.find(id="priceblock_ourprice").get_text()

# The output desired from the page looks ugly so lets clean it
    clean_product_information = product_information.strip()
    clean_product_price = product_price.strip()
    collection_date = datetime.date.today()

# We want to store the data in a csv file. Files have headers along with information
    header = ['Date', 'Description', 'Price']
    data = [collection_date, clean_product_information, clean_product_price]

    file_exist = os.path.exists("Amazon Info.csv")
    while not file_exist:
        with open('Amazon Info.csv', 'w', newline= '', encoding= 'UTF8') as f:
    # with the function open, create the file and then write the data (w) on a newline. (not sure if I need to specify encoding though)
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)
    else:
    # This is our code to append new data to the file.
        with open('Amazon Info.csv', 'a+', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)

# while(True):
#     check_price()
#     time.sleep(5)

start_time = time.time()
seconds = 60*2

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time < seconds:
        check_price()
        time.sleep(5)
    else:
        break


df = pd.read_csv(r'C:\Users\Nicholas\PycharmProjects\Amazon Web Scraping\Amazon Info.csv')
print(df)

def send_mail():
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    # server.starttls()
    server.ehlo()
    server.login('Email address', 'password')

    subject = "The Shirt you want is below $15! Now is your chance to buy!"
    body = "Nicholas, This is the moment we have been waiting for. Now is your chance to pick up the shirt of your dreams. Don't mess it up! Link here: https://www.amazon.com/Funny-Data-Systems-Business-Analyst/dp/B07FNW9FGJ/ref=sr_1_3?dchild=1&keywords=data+analyst+tshirt&qid=1626655184&sr=8-3"

    msg = f"Subject: {subject}\n\n{body}"

    server.sendmail(
        'Nickenney13@gmail.com',
        msg

    )
