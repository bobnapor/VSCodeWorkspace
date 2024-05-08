from selenium import webdriver

# Launch a headless browser
options = webdriver.ChromeOptions()
options.add_argument('headless')  # Run Chrome in headless mode
driver = webdriver.Chrome(options=options)

# Open the webpage
driver.get('https://www.fantasy.top/home')

# Execute JavaScript if needed
driver.execute_script('console.log("This is JavaScript running on the webpage")')

# Now you can scrape the contents
html_content = driver.page_source

# Close the browser
driver.quit()
