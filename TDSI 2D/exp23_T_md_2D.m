function f=exp23_T_md_2D(x_vec,path,lT,xi,yj);
x=path.x0+x_vec*path.cos;
y=path.y0+x_vec*path.sin;
f=exp23B_TT(x,y,xi,yj,lT);