


use criterion::{criterion_group, criterion_main, Criterion, black_box};


use coreset_sc::gen_sbm_with_self_loops;


fn bench_sbm(c: &mut Criterion) {
    let mut group = c.benchmark_group("sbm");

    let ns = [1000];
    let ks = [20];


    for n in ns{
        for k in ks{
            let p = 0.5;
            let q = (1.0/(n as f64))/(k as f64);
            group.sample_size(10);
            group.nresamples(10);
            group.measurement_time(std::time::Duration::from_secs(10));
            group.bench_function(
                format!("gen_sbm_{}_{}",n,k).as_str(),
                |b| b.iter(|| black_box(gen_sbm_with_self_loops(black_box(n),black_box(k),black_box(p),black_box(q)))
            ));

        }
    }
    group.finish();
}



criterion_group!(benches, bench_sbm);
criterion_main!(benches);
