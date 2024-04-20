cl__1 = 0.05;

xmin = DefineNumber[-0.5];
xmax = DefineNumber[2.0];
ymin = DefineNumber[-0.5];
ymax = DefineNumber[1.5];


Point(1) = {xmin, ymin, 0, cl__1};
Point(2) = {xmin, ymax, 0, cl__1};
Point(3) = {xmax, ymax, 0, cl__1};
Point(4) = {xmax, ymin, 0, cl__1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(6) = {4, 1, 2, 3};
Plane Surface(6) = {6};
Physical Surface("Domain", 9) = {6};
Physical Line("Inflow", 1) = {1, 2, 3, 4};
