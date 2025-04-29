x = linspace(0, 50, 50)
a = 2
f = @(x) a*(x-10).^(1/4)
y = f(x)

plot(x, y)

