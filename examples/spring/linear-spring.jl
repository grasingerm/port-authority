using PyPlot;

function time_avg(ts, ys)
  avgs = zeros(length(ts));
  avgs[1] = ys[1];
  sum = 0.0;
  for i=2:length(ts)
    dt = ts[i] - ts[i-1];
    sum += dt * (ys[i] + ys[i-1]) / 2.0;
    avgs[i] = sum / ts[i];
  end
  return avgs;
end

#=
for datafile in filter(x->contains(x, "linear-spring_beta"), readdir("."))
  basename = split(datafile, ',')[1];
  data, headers = readdlm(datafile, ','; header=true);
  ts = vec(data[:,1]);
  Us = vec(data[:,2]);
  xs = vec(data[:,3]);
  xsqs = vec(data[:,4]);

  plot(ts, time_avg(ts, Us));
  xlabel("Number of steps");
  ylabel("\$ \\langle U \\rangle \$");
  savefig(basename * "_U.png");
  clf();

  plot(ts, time_avg(ts, xs));
  xlabel("Number of steps");
  ylabel("\$ \\langle x \\rangle \$");
  savefig(basename * "_x.png");
  clf();

  plot(ts, time_avg(ts, xsqs));
  xlabel("Number of steps");
  ylabel("\$ \\langle x^2 \\rangle \$");
  savefig(basename * "_xsq.png");
  clf();
end
=#

clf();
for (y, ystr, loc) in zip(2:4, ["U", "x", "x^2"], [4, 1, 4])
  for (beta, bstr) in zip([0.1; 1.0; 5.0; 10.0], ["00100"; "01000"; "05000"; "10000"])
    for (delta, dstr) in zip([0.05; 0.25; 1.25; 4.0], ["00050", "00250", "01250", "04000"])
      datafile = "linear-spring_beta-$(bstr)_delta-$(dstr).csv";
      data, headers = readdlm(datafile, ','; header=true);
      ts = vec(data[:,1]);
      ys = vec(data[:,y]);

      plot(ts, time_avg(ts, ys), label="\$\\delta = $delta\$");
    end
    xlabel("Number of steps");
    ylabel("\$ \\langle $ystr \\rangle \$");
    legend(; loc=loc);
    savefig("linear-spring_beta-$(bstr)_$ystr.png");
    clf();
  end
end
