import pandas as pd
df = pd.read_csv("credit_rating.csv")

#Taking the columns that are required
col = []
for column in df.columns:
    col.append(column)
print(col)
no_of_contributing_attributes = len(col)-1

val = []
print("\nEnter values of the attribute for the unknown samples that you want to classify:")
for i in range(0,no_of_contributing_attributes):
    input_value = input(col[i]+":")
    val.append(input_value)
print(val)

# Analysing the hypothesis and calculating the probabilities of each class
# Here,"Buys Computer" is the target wherein the data is classified as "no" and "yes" which is represented as C1 and C2 respectively
classes = []
prob = []
value = []
target_class = col[no_of_contributing_attributes] #a
target_class_count = df.groupby(target_class).size() #p

for i in target_class_count.index:
    classes.append(i)
    
for i in target_class_count:
    value.append(i)
    i=i/len(df)
    prob.append(i)
    

print("\nStep 1: No of Classes (Hypothesis) :", classes)
print(" \nNo of [no,yes]  =", value)
print("\ncalculating class prior probabilites  [P(C1),P(C2)] =", prob)

total_count = []
count_no = []
count_yes = []
b = df[col[no_of_contributing_attributes]].values

for i in df:
    if i==col[no_of_contributing_attributes]:
        break
    a = df[i].values
    #print(a)
    count = 0
    countn = 0
    for k in val:
        q=0
        for j in a:
            q = q+1
            if k==j:
                count=count+1
                if b[q-1]==classes[0]:
                    countn = countn+1
    count_yes.append(count-countn)
    count_no.append(countn)
    total_count.append(count)
print("\n Attribute values of unknown sample: ", val, "\n Yes:",count_yes, "\n NO:",count_no, "\nTotal (C):", total_count)

# Calculating P(X/Ci) for each class 
print("\n Calculating P(X/CI) for each class")
p1 = 1
p2 = 1
p =[]

for i in count_no:
    p1 = p1*(i/int(value[0]))
p.append(p1)


for i in count_yes:
    p2 = p2*(i/int(value[1]))
p.append(p2)

print("\nP(X/C1), P(X/C2) :",p)

ans = []
for i in range(len(prob)):
    for j in range(len(p)):
        if i==j:
            ans.append(prob[i]*p[j])

#calculating posterior probability
            
print("\n Calculating posterior probability")
ans1 = p1*prob[0]
ans2 = p2*prob[1]
print("\n P(C1/X) = {} * {} = {} ".format(p1,prob[0],ans1))
print("\n P(C2/X) = {} * {} = {} ".format(p2,prob[1],ans2))

#conclusion
print(classes)
if(ans1>ans2):
    print("\n Conclusion: Since P(X) is constant and common for both the classes = {} and the maximum posterior is P(C1/X) = {}.Hence unknown sample gets classified as no ".format(classes,ans1))
else:
    print("\n Conclusion: Since P(X) is constant and common for both the classes = {} and the maximum posterior is P(C2/X) = {}.Hence unknown sample gets classified as yes ".format(classes,ans2))

