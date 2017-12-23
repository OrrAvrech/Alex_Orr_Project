function noised = add_noise(source, sigma)

noised = poissrnd(source) + sigma*randn(size(source));

end