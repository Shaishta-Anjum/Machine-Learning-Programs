import csv

# Open the CSV file and read the data
with open('enjoysports.csv', 'r') as file:
    data = [row for row in csv.reader(file)]

# Output the total number of training instances
print("The total number of training instances are:", len(data) - 1)

# Get the number of attributes (excluding the target attribute)
num_attribute = len(data[0]) - 1

# Initialize the hypothesis with '0'
hypothesis = ['0'] * num_attribute

# Process each training instance
for i in range(1, len(data)):  # Start from 1 to skip the header row
    if data[i][num_attribute] == 'yes':
        for j in range(num_attribute):
            if hypothesis[j] == '0' or hypothesis[j] == data[i][j]:
                hypothesis[j] = data[i][j]
            else:
                hypothesis[j] = '?'
        # Output the hypothesis for the current training instance
        print("\nThe hypothesis for the training instance {} is:".format(i), hypothesis)

# Output the final maximally specific hypothesis
print("\nThe Maximally specific hypothesis for the training instances is:", hypothesis)