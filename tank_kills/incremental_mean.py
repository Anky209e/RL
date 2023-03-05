a = [1,2,3,4,5]
b = [1,2,3,4,5,6]
mean_of_a = sum(a)/len(a)
mean_of_b = sum(b)/len(b)

'''
inc-mean = u(k-1) + 1/k[x(k)-u(k-1)]

inc_mean = old_mean  +   [new_element - old_mean]/new_length
'''
print(mean_of_a,mean_of_b)

new_inc = mean_of_a + (6-mean_of_a)/6

print(new_inc)
