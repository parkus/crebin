crebin
======

Code to rapidly and accurately rebin binned data (such as spectra) with flexibility in how values are combined in the binning. 

Normal rebinning takes the average of the values of the bins, weighted by the width of each bin. For example, if a new bin covers 40% of bin a, all of bin b, and 70% of bin c, then the value in that new bin is

(0.4*a + 1.0*b + 0.7*c)/(0.4*w_a + 1.0*w_b + 0.7*w_c)

where a,b, and c are the values in the bins and w_a, w_b, and w_c are the bin widths.

Sometimes, however, you might desire other types of binning. For example, if you are tracking data quality flags, then you will want to "or" the bins. That is, if two bins are combined and one has a data quality flag, you want the new bin to have that flag as well.

As of 2017/05/10 `crebin` supports average (normal), sum, and, or, min, and max rebinning. 

I wrote this for use across my data analysis codes. It isn't pretty and I'm not devoting time to supporting it, but if you're looking for fast and accurate code to handle rebinning data, it might be worth your time to implement this. 


