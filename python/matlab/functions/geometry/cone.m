function z = cone(pn,a,b,c,z_0,x,y)
%- Purpose: Constructs a cone surface of eliptical parameters 'a,b,c' and
%--- z-shift 'z_0' on grid '(x,y)'
%- Note: Flag 'pn' specifies upper (pn=1) or lower (pn=-1) of cone function
z = pn*sqrt(c^2/a^2*x^2 + c^2/b^2*y^2) + z_0;
