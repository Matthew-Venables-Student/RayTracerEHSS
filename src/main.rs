// EHSS-style hard-sphere multiple-scattering ray tracer
// ------------------------------------------------------
// Implements the Monte Carlo trajectory method used to compute
// gas-phase ion collision cross sections, in the spirit of the
// "Exact Hard Spheres Scattering" (EHSS) model of Shvartsburg &
// Jarrold, Chem. Phys. Lett. 261 (1996) 86-91.
//
// Physical picture:
//   - The ion is a rigid cluster of hard spheres (one per atom).
//   - A point-like buffer-gas projectile is fired in on a straight
//     line with impact parameter b, direction fixed.
//   - Each time the trajectory hits a sphere it reflects specularly
//     (elastic hard-sphere collision -> mirror reflection about the
//     surface normal at the contact point). It may bounce between
//     several atoms before escaping to infinity.
//   - The net deflection angle chi(b) between the incoming and
//     outgoing directions is recorded.
//   - Averaging (1 - cos(chi)) over impact parameter (importance
//     sampled as b db) and over random molecular orientations gives
//     the momentum-transfer collision cross section:
//
//         Omega = 2*pi * integral_0^bmax (1 - cos(chi(b))) b db
//


use rand::Rng;
use rayon::prelude::*;
use std::env;
use std::f64::consts::PI;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------
// Minimal 3-vector math
// ---------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }
    fn zero() -> Self {
        Vec3::new(0.0, 0.0, 0.0)
    }
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
    fn normalize(self) -> Vec3 {
        let l = self.length();
        if l < 1e-15 {
            self
        } else {
            self.scale(1.0 / l)
        }
    }
}

// ---------------------------------------------------------------
// A rigid-sphere "atom" in the target molecule
// ---------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct Atom {
    pos: Vec3,
    /// hard-sphere collision radius = atomic vdW-ish radius + buffer-gas radius
    radius: f64,
}

struct Molecule {
    atoms: Vec<Atom>,
}

impl Molecule {
    fn recenter(&mut self) {
        let n = self.atoms.len() as f64;
        let mut c = Vec3::zero();
        for a in &self.atoms {
            c = c.add(a.pos);
        }
        c = c.scale(1.0 / n);
        for a in &mut self.atoms {
            a.pos = a.pos.sub(c);
        }
    }

    /// Apply a rotation quaternion, writing straight into a
    /// Structure-of-Arrays layout instead of building an intermediate
    /// Vec<Atom>. This is the form the hot intersection loop consumes.
    /// Shared by both the random (Monte Carlo) and quasi-random
    /// (low-discrepancy) orientation-generation paths below.
    fn rotate_to_soa(&self, qw: f64, qx: f64, qy: f64, qz: f64) -> MoleculeSoA {
        let mut soa = MoleculeSoA::with_capacity(self.atoms.len());
        for a in &self.atoms {
            let p = rotate_by_quat(a.pos, qw, qx, qy, qz);
            soa.push(p, a.radius);
        }
        soa
    }

    /// Random rotation via Shoemake's uniform-random-quaternion method,
    /// drawing three independent U(0,1) samples from `rng`.
    fn random_rotated_soa<R: Rng>(&self, rng: &mut R) -> MoleculeSoA {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen::<f64>() * 2.0 * PI;
        let u3: f64 = rng.gen::<f64>() * 2.0 * PI;
        let (qw, qx, qy, qz) = quaternion_from_uniforms(u1, u2, u3);
        self.rotate_to_soa(qw, qx, qy, qz)
    }

    /// Deterministic, quasi-random rotation: same Shoemake construction
    /// as `random_rotated_soa`, but the three U(0,1) inputs come from a
    /// Halton low-discrepancy sequence (bases 2, 3, 5) indexed by the
    /// orientation number instead of a pseudorandom generator. The
    /// n-th orientation is always exactly the same rotation, run to
    /// run and machine to machine - no RNG, no seed, nothing random at
    /// all - while still filling SO(3) evenly as n_orientations grows.
    fn quasi_rotated_soa(&self, orientation_index: u64) -> MoleculeSoA {
        let n = orientation_index + 1; // Halton index 0 maps to (0,0,0); start at 1
        let u1 = halton(n, 2);
        let u2 = halton(n, 3) * 2.0 * PI;
        let u3 = halton(n, 5) * 2.0 * PI;
        let (qw, qx, qy, qz) = quaternion_from_uniforms(u1, u2, u3);
        self.rotate_to_soa(qw, qx, qy, qz)
    }

    /// Convert directly to SoA with no rotation applied. Used by tests
    /// and by anything that wants the un-rotated intersection layout.
    #[allow(dead_code)]
    fn to_soa(&self) -> MoleculeSoA {
        let mut soa = MoleculeSoA::with_capacity(self.atoms.len());
        for a in &self.atoms {
            soa.push(a.pos, a.radius);
        }
        soa
    }
}

// ---------------------------------------------------------------
// Structure-of-Arrays molecule layout for the hot intersection loop.
//
// The naive Array-of-Structs (`Vec<Atom>`) forces the compiler to load
// interleaved x/y/z/radius fields with a stride, which defeats
// auto-vectorization: LLVM can't pack four *consecutive* atoms' x
// coordinates into one SIMD register when they're not contiguous in
// memory. Splitting each field into its own flat `Vec<f64>` fixes
// that: `x[i]`, `y[i]`, `z[i]`, `r2[i]` for atoms i, i+1, i+2, i+3 are
// each contiguous, so the ray-sphere test loop below can be
// vectorized (auto-SIMD'd) by the compiler across 2-4 atoms at a
// time on typical x86_64/AArch64 targets, especially when built with
// `RUSTFLAGS="-C target-cpu=native"` to unlock AVX2/NEON codegen for
// your specific machine. The loop body is also written branch-free
// (no early return / no Option per-atom) since branches inside a loop
// are the other big thing that blocks vectorization.
// ---------------------------------------------------------------
struct MoleculeSoA {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    r: Vec<f64>,
    r2: Vec<f64>, // radius^2, precomputed once per orientation instead of per ray
}

impl MoleculeSoA {
    fn with_capacity(n: usize) -> Self {
        MoleculeSoA {
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            z: Vec::with_capacity(n),
            r: Vec::with_capacity(n),
            r2: Vec::with_capacity(n),
        }
    }

    fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
        self.z.clear();
        self.r.clear();
        self.r2.clear();
    }

    fn push(&mut self, pos: Vec3, radius: f64) {
        self.x.push(pos.x);
        self.y.push(pos.y);
        self.z.push(pos.z);
        self.r.push(radius);
        self.r2.push(radius * radius);
    }

    fn len(&self) -> usize {
        self.x.len()
    }

    fn atom_pos(&self, i: usize) -> Vec3 {
        Vec3::new(self.x[i], self.y[i], self.z[i])
    }

    /// Bounding radius from the current centroid-relative frame; used to
    /// pick a safe max impact parameter (bmax) and trajectory start distance.
    fn bounding_radius(&self) -> f64 {
        let mut m = 0.0_f64;
        for i in 0..self.len() {
            let d = (self.x[i] * self.x[i] + self.y[i] * self.y[i] + self.z[i] * self.z[i]).sqrt()
                + self.r[i];
            if d > m {
                m = d;
            }
        }
        m
    }
}

// ---------------------------------------------------------------
// Simple BVH (axis-aligned bounding box) for atom positions in
// the SoA layout. Built per-orientation and used to cull many
// sphere tests for large molecules.
// ---------------------------------------------------------------
#[derive(Debug)]
enum BVH {
    Leaf { bbox_min: Vec3, bbox_max: Vec3, indices: Vec<usize> },
    Node { bbox_min: Vec3, bbox_max: Vec3, left: Box<BVH>, right: Box<BVH> },
}

fn bbox_union(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> (Vec3, Vec3) {
    let min = Vec3::new(a_min.x.min(b_min.x), a_min.y.min(b_min.y), a_min.z.min(b_min.z));
    let max = Vec3::new(a_max.x.max(b_max.x), a_max.y.max(b_max.y), a_max.z.max(b_max.z));
    (min, max)
}

fn compute_bbox_for_indices(soa: &MoleculeSoA, indices: &[usize]) -> (Vec3, Vec3) {
    let mut min = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &i in indices {
        let x = soa.x[i];
        let y = soa.y[i];
        let z = soa.z[i];
        let r = soa.r[i];
        min.x = min.x.min(x - r);
        min.y = min.y.min(y - r);
        min.z = min.z.min(z - r);
        max.x = max.x.max(x + r);
        max.y = max.y.max(y + r);
        max.z = max.z.max(z + r);
    }
    (min, max)
}

fn build_bvh_recursive(soa: &MoleculeSoA, indices: &mut [usize], leaf_size: usize) -> BVH {
    let (bbox_min, bbox_max) = compute_bbox_for_indices(soa, indices);
    if indices.len() <= leaf_size {
        return BVH::Leaf { bbox_min, bbox_max, indices: indices.to_vec() };
    }

    // choose split axis by largest extent
    let ext_x = bbox_max.x - bbox_min.x;
    let ext_y = bbox_max.y - bbox_min.y;
    let ext_z = bbox_max.z - bbox_min.z;
    let axis = if ext_x >= ext_y && ext_x >= ext_z { 0 } else if ext_y >= ext_z { 1 } else { 2 };

    // sort indices by centroid along axis
    indices.sort_by(|&a, &b| {
        let ca = match axis {
            0 => soa.x[a],
            1 => soa.y[a],
            _ => soa.z[a],
        };
        let cb = match axis {
            0 => soa.x[b],
            1 => soa.y[b],
            _ => soa.z[b],
        };
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = indices.len() / 2;
    let (left_idx, right_idx) = indices.split_at_mut(mid);
    let left = build_bvh_recursive(soa, left_idx, leaf_size);
    let right = build_bvh_recursive(soa, right_idx, leaf_size);
    let (lmin, lmax) = match &left {
        BVH::Leaf{bbox_min, bbox_max, ..} | BVH::Node{bbox_min, bbox_max, ..} => (*bbox_min, *bbox_max),
    };
    let (rmin, rmax) = match &right {
        BVH::Leaf{bbox_min, bbox_max, ..} | BVH::Node{bbox_min, bbox_max, ..} => (*bbox_min, *bbox_max),
    };
    let (nmin, nmax) = bbox_union(lmin, lmax, rmin, rmax);

    BVH::Node { bbox_min: nmin, bbox_max: nmax, left: Box::new(left), right: Box::new(right) }
}

fn build_bvh(soa: &MoleculeSoA) -> BVH {
    let n = soa.len();
    let mut indices: Vec<usize> = (0..n).collect();
    if n == 0 {
        // empty leaf
        return BVH::Leaf { bbox_min: Vec3::zero(), bbox_max: Vec3::zero(), indices: Vec::new() };
    }
    build_bvh_recursive(soa, &mut indices, 8)
}

fn ray_aabb_hit(origin: Vec3, dir: Vec3, invdir: Vec3, bbox_min: Vec3, bbox_max: Vec3, t_max: f64) -> bool {
    // slab method
    let mut t0 = (bbox_min.x - origin.x) * invdir.x;
    let mut t1 = (bbox_max.x - origin.x) * invdir.x;
    if invdir.x < 0.0 { std::mem::swap(&mut t0, &mut t1); }
    let mut tmin = t0.max(0.0);
    let mut tmax = t1.min(t_max);

    let mut ty0 = (bbox_min.y - origin.y) * invdir.y;
    let mut ty1 = (bbox_max.y - origin.y) * invdir.y;
    if invdir.y < 0.0 { std::mem::swap(&mut ty0, &mut ty1); }
    tmin = tmin.max(ty0);
    tmax = tmax.min(ty1);
    if tmax < tmin { return false; }

    let mut tz0 = (bbox_min.z - origin.z) * invdir.z;
    let mut tz1 = (bbox_max.z - origin.z) * invdir.z;
    if invdir.z < 0.0 { std::mem::swap(&mut tz0, &mut tz1); }
    tmin = tmin.max(tz0);
    tmax = tmax.min(tz1);
    tmax >= tmin
}

fn nearest_hit_bvh(origin: Vec3, dir: Vec3, mol: &MoleculeSoA, bvh: &BVH, eps: f64) -> Option<(f64, usize, f64)> {
    let invdir = Vec3::new(1.0/dir.x, 1.0/dir.y, 1.0/dir.z);
    let mut best_t = f64::INFINITY;
    let mut best_i = usize::MAX;
    let mut best_disc = 0.0_f64;

    // stack-based traversal
    let mut stack: Vec<&BVH> = Vec::with_capacity(64);
    stack.push(bvh);
    while let Some(node) = stack.pop() {
        match node {
            BVH::Leaf { bbox_min, bbox_max, indices } => {
                if !ray_aabb_hit(origin, dir, invdir, *bbox_min, *bbox_max, best_t) {
                    continue;
                }
                for &i in indices {
                    let ocx = origin.x - mol.x[i];
                    let ocy = origin.y - mol.y[i];
                    let ocz = origin.z - mol.z[i];

                    let b = ocx * dir.x + ocy * dir.y + ocz * dir.z;
                    let c = ocx * ocx + ocy * ocy + ocz * ocz - mol.r2[i];
                    let disc = b * b - c;
                    if disc < 0.0 { continue; }
                    let sq = disc.sqrt();
                    let t1 = -b - sq;
                    let t2 = -b + sq;
                    let mut candidate = f64::INFINITY;
                    if t1 > eps { candidate = t1; } else if t2 > eps { candidate = t2; }
                    if candidate < best_t {
                        best_t = candidate;
                        best_i = i;
                        best_disc = disc;
                    }
                }
            }
            BVH::Node { bbox_min, bbox_max, left, right } => {
                if !ray_aabb_hit(origin, dir, invdir, *bbox_min, *bbox_max, best_t) {
                    continue;
                }
                // push children; order doesn't matter much
                stack.push(left);
                stack.push(right);
            }
        }
    }

    if best_i == usize::MAX { None } else { Some((best_t, best_i, best_disc)) }
}

impl Molecule {
    /// Rotate into an existing SoA buffer to avoid repeated allocations.
    fn rotate_to_soa_into(&self, qw: f64, qx: f64, qy: f64, qz: f64, soa: &mut MoleculeSoA) {
        soa.clear();
        soa.x.reserve(self.atoms.len());
        soa.y.reserve(self.atoms.len());
        soa.z.reserve(self.atoms.len());
        soa.r.reserve(self.atoms.len());
        soa.r2.reserve(self.atoms.len());
        for a in &self.atoms {
            let p = rotate_by_quat(a.pos, qw, qx, qy, qz);
            soa.push(p, a.radius);
        }
    }

    fn random_rotated_soa_into<R: Rng>(&self, rng: &mut R, soa: &mut MoleculeSoA) {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen::<f64>() * 2.0 * PI;
        let u3: f64 = rng.gen::<f64>() * 2.0 * PI;
        let (qw, qx, qy, qz) = quaternion_from_uniforms(u1, u2, u3);
        self.rotate_to_soa_into(qw, qx, qy, qz, soa);
    }

    fn quasi_rotated_soa_into(&self, orientation_index: u64, soa: &mut MoleculeSoA) {
        let n = orientation_index + 1;
        let u1 = halton(n, 2);
        let u2 = halton(n, 3) * 2.0 * PI;
        let u3 = halton(n, 5) * 2.0 * PI;
        let (qw, qx, qy, qz) = quaternion_from_uniforms(u1, u2, u3);
        self.rotate_to_soa_into(qw, qx, qy, qz, soa);
    }
}

fn rotate_by_quat(v: Vec3, qw: f64, qx: f64, qy: f64, qz: f64) -> Vec3 {
    // Standard quaternion-vector rotation: v' = q v q^-1, expanded.
    let (x, y, z) = (v.x, v.y, v.z);
    let (w2, x2, y2, z2) = (qw * qw, qx * qx, qy * qy, qz * qz);
    let xy = qx * qy;
    let xz = qx * qz;
    let yz = qy * qz;
    let wx = qw * qx;
    let wy = qw * qy;
    let wz = qw * qz;

    Vec3::new(
        (w2 + x2 - y2 - z2) * x + 2.0 * (xy - wz) * y + 2.0 * (xz + wy) * z,
        2.0 * (xy + wz) * x + (w2 - x2 + y2 - z2) * y + 2.0 * (yz - wx) * z,
        2.0 * (xz - wy) * x + 2.0 * (yz + wx) * y + (w2 - x2 - y2 + z2) * z,
    )
}

/// Shoemake's uniform-random-rotation construction: given three values
/// u1 in [0,1] and u2,u3 in [0,2*pi], returns a quaternion that is a
/// uniformly-distributed rotation when (u1,u2,u3) are themselves drawn
/// uniformly (or, for quasi-random sampling, drawn from a
/// low-discrepancy sequence covering the same ranges evenly).
fn quaternion_from_uniforms(u1: f64, u2: f64, u3: f64) -> (f64, f64, f64, f64) {
    let s1 = (1.0 - u1).sqrt();
    let s2 = u1.sqrt();
    let qw = s1 * u2.sin();
    let qx = s1 * u2.cos();
    let qy = s2 * u3.sin();
    let qz = s2 * u3.cos();
    (qw, qx, qy, qz)
}

/// The Halton sequence: a deterministic low-discrepancy sequence in
/// [0,1), one of the standard building blocks of quasi-Monte Carlo
/// integration. For a fixed prime `base`, `halton(1, base)`,
/// `halton(2, base)`, `halton(3, base)`, ... fills [0,1) far more
/// evenly than a pseudorandom sequence of the same length (no random
/// clumping or gaps), which is what lets quasi-Monte Carlo integration
/// converge faster than plain Monte Carlo for a given sample count -
/// and, crucially here, makes every run 100% reproducible with no RNG
/// or seed involved at all. Different dimensions of a multi-dimensional
/// QMC point should use different (coprime, conventionally prime)
/// bases, which is why orientations use bases 2/3/5 and impact
/// parameters use a separate base (7) below.
fn halton(mut index: u64, base: u64) -> f64 {
    let mut result = 0.0_f64;
    let mut f = 1.0_f64;
    while index > 0 {
        f /= base as f64;
        result += f * (index % base) as f64;
        index /= base;
    }
    result
}

// ---------------------------------------------------------------
// Ray - sphere intersection
// ---------------------------------------------------------------
/// Returns the smallest positive t (beyond eps) at which the ray
/// origin + t*dir hits the sphere, if any. Kept as a simple scalar
/// reference implementation used by tests to cross-check the
/// vectorized `nearest_hit` below.
#[allow(dead_code)]
fn ray_sphere_hit(origin: Vec3, dir: Vec3, center: Vec3, radius: f64, eps: f64) -> Option<f64> {
    let oc = origin.sub(center);
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t1 = -b - sq;
    if t1 > eps {
        return Some(t1);
    }
    let t2 = -b + sq;
    if t2 > eps {
        return Some(t2);
    }
    None
}

/// Find the nearest atom the ray hits, if any: returns (t, atom_index).
///
/// Written branch-free inside the loop (no early `return`, no
/// per-atom `Option`) and over flat `&[f64]` slices so LLVM's
/// auto-vectorizer has the best chance of packing this into SIMD
/// instructions across several atoms at once. The reduction
/// (tracking the running minimum t and its index) still needs a
/// compare per iteration, but that's cheap and vectorizes fine too
/// (packed-compare + blend).
#[inline]
fn nearest_hit(origin: Vec3, dir: Vec3, mol: &MoleculeSoA, eps: f64) -> Option<(f64, usize, f64)> {
    let (ox, oy, oz) = (origin.x, origin.y, origin.z);
    let (dx, dy, dz) = (dir.x, dir.y, dir.z);
    let mut best_t = f64::INFINITY;
    let mut best_i = usize::MAX;
    let mut best_disc = 0.0_f64;

    let n = mol.len();
    for i in 0..n {
        let ocx = ox - mol.x[i];
        let ocy = oy - mol.y[i];
        let ocz = oz - mol.z[i];

        let b = ocx * dx + ocy * dy + ocz * dz;
        let c = ocx * ocx + ocy * ocy + ocz * ocz - mol.r2[i];
        let disc = b * b - c;

        // Branch-free candidate t: if disc < 0 (miss), sqrt(disc.max(0))
        // is 0 and we rely on the eps/INFINITY selects below to discard
        // it rather than branching on `disc < 0.0` directly.
        let sq = disc.max(0.0).sqrt();
        let hit = disc >= 0.0;

        let t1 = -b - sq;
        let t2 = -b + sq;
        let t1_valid = hit & (t1 > eps);
        let t2_valid = hit & (t2 > eps);

        let candidate = if t1_valid {
            t1
        } else if t2_valid {
            t2
        } else {
            f64::INFINITY
        };

        if candidate < best_t {
            best_t = candidate;
            best_i = i;
            best_disc = disc;
        }
    }

    if best_i == usize::MAX {
        None
    } else {
        Some((best_t, best_i, best_disc))
    }
}

// ---------------------------------------------------------------
// Trajectory tracing: fire a projectile at impact parameter b,
// bounce it (specular reflection) off hard spheres until it escapes,
// and return cos(chi), the cosine of the net deflection angle.
// ---------------------------------------------------------------
struct TraceParams {
    max_bounces: usize,
}

fn trace_trajectory(mol: &MoleculeSoA, b: f64, start_z: f64, params: &TraceParams, eps: f64, bvh: Option<&BVH>) -> f64 {
    let dir0 = Vec3::new(0.0, 0.0, -1.0);
    let mut origin = Vec3::new(b, 0.0, start_z);
    let mut dir = dir0;
    let mut bounce = 0usize;
    // recent hit history for cycle detection (small ring buffer)
    let mut recent_hits: Vec<usize> = Vec::with_capacity(32);
    
    for _ in 0..params.max_bounces {
        let hit = match bvh {
            Some(tree) => nearest_hit_bvh(origin, dir, mol, tree, eps),
            None => nearest_hit(origin, dir, mol, eps),
        };
        match hit {
            None => break, // escaped to infinity
            Some((t, idx, disc)) => {
                // cycle detection: push recent hit and evict oldest when full
                recent_hits.push(idx);
                if recent_hits.len() > 32 { recent_hits.remove(0); }
                // if the same atom appears many times recently, assume stuck
                let repeats = recent_hits.iter().filter(|&&x| x == idx).count();
                if repeats >= 5 {
                    // stuck: bail out to avoid infinite bouncing
                    break;
                }
                // detect simple two-atom alternation A,B,A,B pattern
                if recent_hits.len() >= 4 {
                    let len = recent_hits.len();
                    if recent_hits[len-1] == recent_hits[len-3] && recent_hits[len-2] == recent_hits[len-4] && recent_hits[len-1] != recent_hits[len-2] {
                        // alternating A,B,A,B -> stuck, bail out
                        break;
                    }
                }
                let hit_point = origin.add(dir.scale(t));
                let atom_pos = mol.atom_pos(idx);

                // grazing/tangent handling: when discriminant is very small
                // the collision is nearly tangent and produces a minimal
                // deflection. In that case, skip the costly bounce and
                // continue the incoming direction to avoid numeric noise.
                let grazing_disc_tol = (eps * 100.0).powi(2); // length^2 threshold
                if disc.abs() <= grazing_disc_tol {
                    let normal = hit_point.sub(atom_pos).normalize();
                    let new_dir = dir.sub(normal.scale(2.0 * dir.dot(normal))).normalize();
                    let cos_chi = dir0.dot(new_dir);
                    let grazing_deflection_thresh = 1e-6_f64; // 1 - cos_chi threshold
                    if (1.0 - cos_chi) < grazing_deflection_thresh {
                        // treat as no meaningful deflection: advance origin
                        origin = hit_point.add(dir.scale(eps * 10.0));
                        continue;
                    }
                    // else fall through to normal reflection handling
                }

                let normal = hit_point.sub(atom_pos).normalize();
                // specular reflection: d' = d - 2 (d.n) n
                let d_dot_n = dir.dot(normal);
                let mut new_dir = dir.sub(normal.scale(2.0 * d_dot_n));
                bounce += 1;
                if bounce % 8 == 0 {
                    new_dir = new_dir.normalize();
                } else {
                    let len2 = new_dir.dot(new_dir);
                    if (len2 - 1.0).abs() > 1e-6 {
                        new_dir = new_dir.normalize();
                    }
                }
                // push origin slightly off the surface along the new
                // direction to avoid re-detecting the same intersection
                origin = hit_point.add(new_dir.scale(eps * 10.0));
                dir = new_dir;
            }
        }
    }
    dir0.dot(dir) // = cos(chi); if the ray never hit anything this is 1.0 (chi = 0)
}

// ---------------------------------------------------------------
// Integration over impact parameter (importance sampled as b db,
// i.e. b = bmax*sqrt(u)) and over molecular orientation, either by
// plain Monte Carlo (pseudorandom) or quasi-Monte Carlo (deterministic
// low-discrepancy Halton sequence).
// ---------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sampling {
    /// Pseudorandom orientations and impact parameters (the original
    /// behaviour). Non-deterministic from run to run unless seeded.
    Random,
    /// Deterministic low-discrepancy (Halton) sequence for both
    /// orientation and impact parameter. Identical output every run,
    /// no RNG or seed involved, and typically lower integration error
    /// than Random for the same sample count.
    Quasi,
}

struct McResult {
    cross_section: f64, // Angstrom^2
    std_error: f64,     // Angstrom^2, see field docs on report_result for caveats
    n_orientations: usize,
    n_impact_per_orientation: usize,
    sampling: Sampling,
}

fn compute_cross_section(
    mol: &Molecule,
    n_orientations: usize,
    n_impact: usize,
    projectile_radius: f64,
    margin: f64,
    sampling: Sampling,
) -> McResult {
    let params = TraceParams { max_bounces: 64 };
    // adaptive eps parameters: local_eps = max(min_eps, rel_eps * bmax)
    let rel_eps = 1e-9_f64;
    let min_eps = 1e-10_f64;

    // Each orientation's trajectories are independent of every other
    // orientation's, so this is embarrassingly parallel either way:
    // hand the work out across all available CPU cores with rayon.
    let per_orientation_q: Vec<f64> = match sampling {
        Sampling::Random => (0..n_orientations)
            .into_par_iter()
            .map_init(
                || (rand::thread_rng(), MoleculeSoA::with_capacity(mol.atoms.len())),
                |state, _| {
                    // state is &mut (rng, soa)
                    let (rng, soa) = state;
                    // draw a rotation into the per-thread soa buffer
                    soa.clear();
                    let u1: f64 = rng.gen();
                    let u2: f64 = rng.gen::<f64>() * 2.0 * PI;
                    let u3: f64 = rng.gen::<f64>() * 2.0 * PI;
                    let (qw, qx, qy, qz) = quaternion_from_uniforms(u1, u2, u3);
                    mol.rotate_to_soa_into(qw, qx, qy, qz, soa);

                    let bound_r = soa.bounding_radius() + projectile_radius;
                    let bmax = bound_r + margin;
                    let start_z = bmax + margin;

                    let mut sum = 0.0_f64;
                    // build BVH for this rotated orientation once
                    let bvh = build_bvh(soa);
                    // adaptive eps for this orientation
                    let local_eps = (rel_eps * bmax).max(min_eps);
                    for _ in 0..n_impact {
                        let u: f64 = rng.gen();
                        let b = bmax * u.sqrt(); // b db importance sampling
                        let cos_chi = trace_trajectory(soa, b, start_z, &params, local_eps, Some(&bvh));
                        sum += 1.0 - cos_chi;
                    }
                    let mean = sum / n_impact as f64;
                    PI * bmax * bmax * mean // Angstrom^2
                },
            )
            .collect(),

        Sampling::Quasi => (0..n_orientations)
            .into_par_iter()
            .map_init(|| MoleculeSoA::with_capacity(mol.atoms.len()), |soa, i| {
                mol.quasi_rotated_soa_into(i as u64, soa);
                let bound_r = soa.bounding_radius() + projectile_radius;
                let bmax = bound_r + margin;
                let start_z = bmax + margin;

                let mut sum = 0.0_f64;
                let bvh = build_bvh(soa);
                let local_eps = (rel_eps * bmax).max(min_eps);
                for j in 0..n_impact {
                    // Flat index over the whole (orientation, impact)
                    // grid, base 7 (distinct from the 2/3/5 used for
                    // orientation), so every one of the
                    // n_orientations * n_impact trajectories in the
                    // whole run gets its own unique, well-spread point
                    // in the Halton sequence - not just within one
                    // orientation's own impact-parameter sweep.
                    let flat = (i * n_impact + j) as u64 + 1;
                    let u = halton(flat, 7);
                    let b = bmax * u.sqrt(); // b db importance sampling
                    let cos_chi = trace_trajectory(soa, b, start_z, &params, local_eps, Some(&bvh));
                    sum += 1.0 - cos_chi;
                }
                let mean = sum / n_impact as f64;
                PI * bmax * bmax * mean // Angstrom^2
            })
            .collect(),
    };

    let n = per_orientation_q.len() as f64;
    let mean_q = per_orientation_q.iter().sum::<f64>() / n;
    let var = per_orientation_q
        .iter()
        .map(|q| (q - mean_q).powi(2))
        .sum::<f64>()
        / (n - 1.0).max(1.0);
    let std_error = (var / n).sqrt();

    McResult {
        cross_section: mean_q,
        std_error,
        n_orientations,
        n_impact_per_orientation: n_impact,
        sampling,
    }
}

// ---------------------------------------------------------------
// Molecule loading: simple XYZ format, plus a tiny built-in
// element -> hard-sphere radius table (ILLUSTRATIVE defaults).
// ---------------------------------------------------------------
fn default_radius_for_element(sym: &str) -> f64 {
    // Illustrative atomic "hard sphere" radii in Angstrom
    // Using the optimised parameters from Sui et al. (2009)
    match sym.to_ascii_uppercase().as_str() {
        "H" => 1.50,
        "C" => 2.70,
        "N" => 2.50,
        "O" => 2.50,
        "F" => 2.50,
        "S" => 2.80,
        "CL" => 2.75,
        "P" => 2.80,
        "NA" => 3.27,
        "K" => 3.75,
        "FE" => 3.00,
        _ => 2.50,
    }
}

fn parse_xyz(contents: &str, projectile_radius: f64) -> Molecule {
    let mut lines = contents.lines();
    let n_atoms: usize = lines
        .next()
        .expect("empty xyz file")
        .trim()
        .parse()
        .expect("first line of xyz must be atom count");
    let _comment = lines.next();

    let mut atoms = Vec::with_capacity(n_atoms);
    for line in lines.take(n_atoms) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }
        let sym = parts[0];
        let x: f64 = parts[1].parse().unwrap();
        let y: f64 = parts[2].parse().unwrap();
        let z: f64 = parts[3].parse().unwrap();
        let r = default_radius_for_element(sym) + projectile_radius;
        atoms.push(Atom {
            pos: Vec3::new(x, y, z),
            radius: r,
        });
    }
    Molecule { atoms }
}

/// Guess an element symbol from a PDB atom name when columns 77-78
/// (the official element field) are blank/missing, which happens in
/// a lot of older or hand-edited PDB files. PDB atom names are
/// right-justified oddly (e.g. " CA " = alpha carbon, "HG21" =
/// hydrogen), so: strip digits, strip leading whitespace, and take
/// the first one or two letters, preferring a known two-letter
/// element only when the name doesn't look like a common biomolecule
/// atom (C/N/O/S/P/H + digits, which is the overwhelming majority
/// case in protein/nucleic-acid PDB files).
fn element_from_atom_name(atom_name: &str) -> String {
    let trimmed = atom_name.trim();
    let letters: String = trimmed
        .chars()
        .take_while(|c| c.is_alphabetic())
        .collect::<String>()
        .to_ascii_uppercase();
    if letters.is_empty() {
        return "C".to_string(); // last-resort fallback
    }

    // A handful of two-letter element codes that show up as PDB atom
    // names almost exclusively as real hetero-ions (not as standard
    // protein/nucleic-acid backbone or sidechain atom labels), so it's
    // safe to read them literally. Deliberately EXCLUDES "CA" and "CD",
    // which are far more commonly alpha-carbon / sidechain delta-carbon
    // labels than calcium/cadmium - those fall through to the
    // single-letter rule below instead.
    let unambiguous_two_letter = [
        "CL", "FE", "ZN", "MG", "BR", "NA", "SE", "MN", "CU", "NI", "CO",
    ];
    if letters.len() >= 2 && unambiguous_two_letter.contains(&letters[..2].as_ref()) {
        return letters[..2].to_string();
    }

    let one = letters[..1].to_string();
    let common_single = ["C", "N", "O", "S", "P", "H"];
    if common_single.contains(&one.as_str()) {
        one
    } else if letters.len() >= 2 {
        letters[..2].to_string()
    } else {
        one
    }
}

fn parse_pdb(contents: &str, projectile_radius: f64) -> Molecule {
    let mut atoms = Vec::new();
    for line in contents.lines() {
        if line.len() < 6 {
            continue;
        }
        let record = &line[0..6];
        if record.trim() != "ATOM" && record.trim() != "HETATM" {
            continue;
        }
        // Fixed PDB columns (1-indexed in the spec -> 0-indexed slices here).
        // 31-38: x, 39-46: y, 47-54: z, 77-78: element symbol.
        let get_slice = |start: usize, end: usize| -> Option<&str> {
            if line.len() >= end {
                Some(line[start..end].trim())
            } else {
                None
            }
        };
        let x: f64 = match get_slice(30, 38).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue, // malformed line, skip
        };
        let y: f64 = match get_slice(38, 46).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let z: f64 = match get_slice(46, 54).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };

        let element = get_slice(76, 78)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                // fall back to atom name field, columns 13-16
                let atom_name = get_slice(12, 16).unwrap_or("C");
                element_from_atom_name(atom_name)
            });

        let r = default_radius_for_element(&element) + projectile_radius;
        atoms.push(Atom {
            pos: Vec3::new(x, y, z),
            radius: r,
        });
    }

    if atoms.is_empty() {
        panic!("no ATOM/HETATM records with parsable coordinates found in PDB file");
    }
    Molecule { atoms }
}

/// A small built-in demo molecule (methane, CH4, roughly tetrahedral,
/// bond length ~1.09 A) so the program is runnable with no input file.
fn demo_methane(projectile_radius: f64) -> Molecule {
    let bl = 1.09_f64;
    let a = bl / 3.0_f64.sqrt();
    let coords = [
        ("C", 0.0, 0.0, 0.0),
        ("H", a, a, a),
        ("H", a, -a, -a),
        ("H", -a, a, -a),
        ("H", -a, -a, a),
    ];
    let atoms = coords
        .iter()
        .map(|(sym, x, y, z)| Atom {
            pos: Vec3::new(*x, *y, *z),
            radius: default_radius_for_element(sym) + projectile_radius,
        })
        .collect();
    Molecule { atoms }
}

// ---------------------------------------------------------------
// main
// ---------------------------------------------------------------
fn print_usage() {
    eprintln!(
        "EHSS-style hard-sphere ray-tracing collision cross section\n\n\
         Usage:\n  ehss_raytracer [FILE.xyz | FILE.pdb] [OPTIONS]\n\n\
         Accepts a simple XYZ file or a PDB file (ATOM/HETATM records; element\n\
         is read from PDB columns 77-78, falling back to the atom name if blank).\n\
         If no file is given, a built-in methane molecule is used.\n\n\
         Options:\n\
         \x20 --orientations N        number of orientations to average over (default 300)\n\
         \x20 --impacts N             impact-parameter samples per orientation (default 400)\n\
         \x20 --projectile-radius R   buffer-gas hard-sphere radius in Angstrom (default 1.00, ~He)\n\
         \x20 --margin R              extra clearance beyond bounding sphere in Angstrom (default 3.0)\n\
         \x20 --sampling MODE         'quasi' (default): deterministic low-discrepancy (Halton)\n\
         \x20                         orientations/impact parameters - same result every run.\n\
         \x20                         'random': pseudorandom Monte Carlo (the original behaviour).\n"
    );
}

fn get_flag_str<'a>(args: &'a [String], name: &str, default: &'a str) -> &'a str {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or(default)
}

fn get_flag_f64(args: &[String], name: &str, default: f64) -> f64 {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

fn get_flag_usize(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return;
    }

    let n_orientations = get_flag_usize(&args, "--orientations", 300);
    let n_impacts = get_flag_usize(&args, "--impacts", 400);
    let projectile_radius = get_flag_f64(&args, "--projectile-radius", 1.00);
    let margin = get_flag_f64(&args, "--margin", 3.0);
    let sampling = match get_flag_str(&args, "--sampling", "quasi") {
        "random" => Sampling::Random,
        "quasi" => Sampling::Quasi,
        other => {
            eprintln!(
                "Unknown --sampling '{}' (expected 'quasi' or 'random'); using 'quasi'.",
                other
            );
            Sampling::Quasi
        }
    };

    let input_path = args
        .iter()
        .find(|a| a.ends_with(".xyz") || a.ends_with(".pdb") || a.ends_with(".ent"));

    let mut mol = match input_path {
        Some(path) => {
            let contents = fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("could not read {}: {}", path, e));
            if path.ends_with(".pdb") || path.ends_with(".ent") {
                parse_pdb(&contents, projectile_radius)
            } else {
                parse_xyz(&contents, projectile_radius)
            }
        }
        None => {
            eprintln!("No .xyz/.pdb file given - using built-in methane (CH4) demo molecule.\n");
            demo_methane(projectile_radius)
        }
    };
    mol.recenter();

    eprintln!(
        "Molecule: {} atoms | projectile radius = {:.3} A | orientations = {} | impacts/orientation = {} | sampling = {:?}",
        mol.atoms.len(),
        projectile_radius,
        n_orientations,
        n_impacts,
        sampling
    );

    let start = Instant::now();
    let result = compute_cross_section(
        &mol,
        n_orientations,
        n_impacts,
        projectile_radius,
        margin,
        sampling,
    );
    let elapsed = start.elapsed();

    println!("\n=== EHSS-style ray-traced collision cross section ===");
    match result.sampling {
        Sampling::Random => {
            println!(
                "Omega = {:.3} +/- {:.3} A^2   (N_orient={}, N_impact={}, sampling=random)",
                result.cross_section,
                result.std_error,
                result.n_orientations,
                result.n_impact_per_orientation
            );
            println!(
                "        = {:.3} +/- {:.3} nm^2",
                result.cross_section / 100.0,
                result.std_error / 100.0
            );
        }
        Sampling::Quasi => {
            // With a deterministic low-discrepancy sequence the
            // per-orientation Q values aren't i.i.d. random samples, so
            // their spread isn't a classical statistical standard
            // error - it's still a useful measure of how much Q varies
            // orientation-to-orientation, just labelled honestly.
            println!(
                "Omega = {:.3} A^2   (N_orient={}, N_impact={}, sampling=quasi/deterministic)",
                result.cross_section, result.n_orientations, result.n_impact_per_orientation
            );
            println!("        = {:.3} nm^2", result.cross_section / 100.0);
            println!(
                "        orientation-to-orientation spread: {:.3} A^2 (not an i.i.d. std error - \
                 see notes)",
                result.std_error
            );
        }
    }
    report_timing(elapsed, n_orientations, n_impacts);
}

/// Report how long the Monte Carlo cross-section calculation took, and
/// the resulting trajectory throughput. Split into its own function so
/// the timing/reporting logic is easy to find and reuse (e.g. if you
/// later want to time individual stages separately).
fn report_timing(elapsed: std::time::Duration, n_orientations: usize, n_impacts: usize) {
    let total_trajectories = n_orientations * n_impacts;
    let secs = elapsed.as_secs_f64();
    let per_traj_us = if total_trajectories > 0 {
        (secs * 1_000_000.0) / total_trajectories as f64
    } else {
        0.0
    };

    println!("\n--- Timing ---");
    if secs < 1.0 {
        println!("Elapsed: {:.2} ms", secs * 1000.0);
    } else {
        println!("Elapsed: {:.3} s", secs);
    }
    println!(
        "Trajectories traced: {} ({:.2} us/trajectory, {:.0} trajectories/s, {} threads)",
        total_trajectories,
        per_traj_us,
        if secs > 0.0 {
            total_trajectories as f64 / secs
        } else {
            0.0
        },
        rayon::current_num_threads()
    );
}

// ---------------------------------------------------------------
// Tests: sanity-check the ray tracer against an analytic case.
// A single isolated hard sphere of radius R has an EXACT
// hard-sphere cross section of pi*R^2.
// ---------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_sphere_matches_pi_r_squared() {
        let mol = Molecule {
            atoms: vec![Atom {
                pos: Vec3::zero(),
                radius: 2.0,
            }],
        };
        let expected = PI * 2.0_f64.powi(2);

        let result_random = compute_cross_section(&mol, 1, 200_000, 0.0, 5.0, Sampling::Random);
        let rel_err_random = (result_random.cross_section - expected).abs() / expected;
        assert!(
            rel_err_random < 0.02,
            "random: got {:.4}, expected {:.4}, rel_err {:.4}",
            result_random.cross_section,
            expected,
            rel_err_random
        );

        let result_quasi = compute_cross_section(&mol, 1, 200_000, 0.0, 5.0, Sampling::Quasi);
        let rel_err_quasi = (result_quasi.cross_section - expected).abs() / expected;
        assert!(
            rel_err_quasi < 0.02,
            "quasi: got {:.4}, expected {:.4}, rel_err {:.4}",
            result_quasi.cross_section,
            expected,
            rel_err_quasi
        );
    }

    #[test]
    fn quasi_sampling_is_deterministic_across_runs() {
        // The whole point of quasi sampling: identical inputs give
        // bit-identical output, run after run, with no RNG involved.
        let mol = Molecule {
            atoms: vec![
                Atom { pos: Vec3::new(0.0, 0.0, 0.0), radius: 1.5 },
                Atom { pos: Vec3::new(2.0, 0.0, 0.0), radius: 1.2 },
            ],
        };
        let r1 = compute_cross_section(&mol, 50, 60, 1.0, 3.0, Sampling::Quasi);
        let r2 = compute_cross_section(&mol, 50, 60, 1.0, 3.0, Sampling::Quasi);
        assert_eq!(r1.cross_section.to_bits(), r2.cross_section.to_bits());
    }

    #[test]
    fn halton_sequence_fills_unit_interval_evenly() {
        // Basic sanity: base-2 Halton values for indices 1..=8 should
        // be the classic {1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, 1/16}
        // van der Corput sequence.
        let expected = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625];
        for (i, exp) in expected.iter().enumerate() {
            let got = halton((i + 1) as u64, 2);
            assert!((got - exp).abs() < 1e-12, "index {}: got {}, expected {}", i + 1, got, exp);
        }
    }

    #[test]
    fn parses_pdb_atom_records() {
        let pdb = "\
HEADER    TEST\n\
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  N1  LIG A   1       1.500   0.000   0.000  1.00  0.00           N\n\
HETATM    3 NA    NA A   2       0.000   3.000   0.000  1.00  0.00          NA\n\
END\n";
        let mol = parse_pdb(pdb, 1.0);
        assert_eq!(mol.atoms.len(), 3);
        assert!((mol.atoms[0].pos.x - 0.0).abs() < 1e-9);
        assert!((mol.atoms[1].pos.x - 1.5).abs() < 1e-9);
        assert!((mol.atoms[2].pos.y - 3.0).abs() < 1e-9);
        // NA (sodium, r=2.27) should give a bigger radius than N (r=1.55)
        assert!(mol.atoms[2].radius > mol.atoms[1].radius);
    }

    #[test]
    fn element_from_name_prefers_biomolecule_singles() {
        assert_eq!(element_from_atom_name("CA"), "C"); // alpha carbon, not calcium
        assert_eq!(element_from_atom_name("CL"), "CL"); // chlorine, not a CA-style name
        assert_eq!(element_from_atom_name("HG21"), "H"); // hydrogen, not mercury
        assert_eq!(element_from_atom_name("OXT"), "O");
    }

    #[test]
    fn ray_sphere_hit_basic() {
        let hit = ray_sphere_hit(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::zero(),
            2.0,
            1e-9,
        );
        assert!(hit.is_some());
        assert!((hit.unwrap() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn specular_reflection_off_single_sphere_at_grazing_b() {
        // b close to R should deflect only slightly (grazing hit).
        let mol = Molecule {
            atoms: vec![Atom {
                pos: Vec3::zero(),
                radius: 1.0,
            }],
        };
        let soa = mol.to_soa();
        let params = TraceParams { max_bounces: 8 };
        let bvh = build_bvh(&soa);
        let local_eps = 1e-9_f64;
        let cos_chi = trace_trajectory(&soa, 0.99, 10.0, &params, local_eps, Some(&bvh));
        assert!(cos_chi > 0.5, "grazing hit should barely deflect");
    }

    #[test]
    fn soa_nearest_hit_matches_scalar_ray_sphere_hit() {
        // Cross-check the vectorization-friendly SoA nearest_hit against
        // the simple scalar ray_sphere_hit it was derived from, across a
        // handful of atoms and both hit/miss rays.
        let mol = Molecule {
            atoms: vec![
                Atom { pos: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 },
                Atom { pos: Vec3::new(5.0, 0.0, 0.0), radius: 1.0 },
                Atom { pos: Vec3::new(0.0, 5.0, 0.0), radius: 1.0 },
            ],
        };
        let soa = mol.to_soa();
        let origin = Vec3::new(0.0, 0.0, 10.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let eps = 1e-9;

        let expected = mol
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(i, a)| ray_sphere_hit(origin, dir, a.pos, a.radius, eps).map(|t| (t, i)))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let got = nearest_hit(origin, dir, &soa, eps);
        assert_eq!(got.map(|(_, i, _)| i), expected.map(|(_, i)| i));
        if let (Some((t1, _, _)), Some((t2, _))) = (got, expected) {
            assert!((t1 - t2).abs() < 1e-9);
        }

        // A ray that misses everything should return None from both.
        let miss_dir = Vec3::new(0.0, 0.0, 1.0); // pointing away from all atoms
        assert!(nearest_hit(origin, miss_dir, &soa, eps).is_none());
    }
}
