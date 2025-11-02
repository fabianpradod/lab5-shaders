use std::error::Error;
use std::f32::consts::FRAC_PI_4;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::render::WindowCanvas;

const SCREEN_WIDTH: i32 = 960;
const SCREEN_HEIGHT: i32 = 720;
const STAR_OBJ_PATH: &str = "star.obj";
const ROCKY_OBJ_PATH: &str = "moon.obj";
const GAS_GIANT_OBJ_PATH: &str = "ring.obj";
const BASE_TILT_X: f32 = -1.05;
const BASE_ROTATION_Y: f32 = -FRAC_PI_4;

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn length_squared(self) -> f32 {
        self.dot(self)
    }

    fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-6 {
            self / len
        } else {
            self
        }
    }

    fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    fn mul_by_vec(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }

    fn clamp01(self) -> Self {
        Self::new(
            self.x.clamp(0.0, 1.0),
            self.y.clamp(0.0, 1.0),
            self.z.clamp(0.0, 1.0),
        )
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProjectedVertex {
    x: f32,
    y: f32,
    z: f32,
}

struct Mesh {
    vertices: Vec<Vec3>,
    faces: Vec<[usize; 3]>,
    vertex_normals: Vec<Vec3>,
}

impl Mesh {
    fn load_from_obj(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut parts = trimmed.split_whitespace();
            match parts.next() {
                Some("v") => {
                    let x: f32 = parts.next().unwrap_or("0.0").parse()?;
                    let y: f32 = parts.next().unwrap_or("0.0").parse()?;
                    let z: f32 = parts.next().unwrap_or("0.0").parse()?;
                    vertices.push(Vec3::new(x, y, z));
                }
                Some("f") => {
                    let mut indices = Vec::new();
                    for token in parts {
                        let index_str = token.split('/').next().unwrap_or("");
                        if index_str.is_empty() {
                            continue;
                        }
                        let index: usize = index_str.parse()?;
                        if index == 0 {
                            continue;
                        }
                        indices.push(index - 1);
                    }

                    if indices.len() >= 3 {
                        for i in 1..indices.len() - 1 {
                            faces.push([indices[0], indices[i], indices[i + 1]]);
                        }
                    }
                }
                _ => continue,
            }
        }

        if vertices.is_empty() {
            return Err("OBJ file does not contain vertices".into());
        }

        let mut mesh = Mesh {
            vertices,
            faces,
            vertex_normals: Vec::new(),
        };
        mesh.normalize_vertices();
        mesh.compute_vertex_normals();
        Ok(mesh)
    }

    fn normalize_vertices(&mut self) {
        if self.vertices.is_empty() {
            return;
        }

        let mut min_x = self.vertices[0].x;
        let mut max_x = self.vertices[0].x;
        let mut min_y = self.vertices[0].y;
        let mut max_y = self.vertices[0].y;
        let mut min_z = self.vertices[0].z;
        let mut max_z = self.vertices[0].z;

        for v in &self.vertices {
            min_x = min_x.min(v.x);
            max_x = max_x.max(v.x);
            min_y = min_y.min(v.y);
            max_y = max_y.max(v.y);
            min_z = min_z.min(v.z);
            max_z = max_z.max(v.z);
        }

        let center = Vec3::new(
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
        );

        let mut max_extent = 0.0f32;
        for v in &self.vertices {
            let offset = *v - center;
            max_extent = max_extent
                .max(offset.x.abs())
                .max(offset.y.abs())
                .max(offset.z.abs());
        }

        if max_extent < 1e-6 {
            max_extent = 1.0;
        }

        for v in &mut self.vertices {
            *v = (*v - center) / max_extent;
        }
    }

    fn compute_vertex_normals(&mut self) {
        self.vertex_normals = vec![Vec3::default(); self.vertices.len()];

        for face in &self.faces {
            let v0 = self.vertices[face[0]];
            let v1 = self.vertices[face[1]];
            let v2 = self.vertices[face[2]];
            let normal = (v1 - v0).cross(v2 - v0);

            self.vertex_normals[face[0]] += normal;
            self.vertex_normals[face[1]] += normal;
            self.vertex_normals[face[2]] += normal;
        }

        for normal in &mut self.vertex_normals {
            *normal = normal.normalize();
        }
    }
}

struct FrameBuffer {
    width: i32,
    height: i32,
    pixels: Vec<Color>,
    depth: Vec<f32>,
}

impl FrameBuffer {
    fn new(width: i32, height: i32) -> Self {
        let pixel_count = (width * height) as usize;
        Self {
            width,
            height,
            pixels: vec![Color::RGB(0, 0, 0); pixel_count],
            depth: vec![f32::INFINITY; pixel_count],
        }
    }

    fn clear(&mut self, color: Color) {
        for pixel in &mut self.pixels {
            *pixel = color;
        }
        for depth in &mut self.depth {
            *depth = f32::INFINITY;
        }
    }

    fn try_set_pixel(&mut self, x: i32, y: i32, color: Color, depth: f32) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }

        let index = (y * self.width + x) as usize;
        if depth < self.depth[index] {
            self.depth[index] = depth;
            self.pixels[index] = color;
        }
    }

    fn draw_to_canvas(&self, canvas: &mut WindowCanvas) -> Result<(), String> {
        for y in 0..self.height {
            for x in 0..self.width {
                let index = (y * self.width + x) as usize;
                canvas.set_draw_color(self.pixels[index]);
                canvas.draw_point(Point::new(x, y))?;
            }
        }
        Ok(())
    }

    fn save_png(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        use std::io::BufWriter;

        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        let mut encoder = png::Encoder::new(&mut writer, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut png_writer = encoder.write_header()?;

        let mut rgb_data = Vec::with_capacity(self.pixels.len() * 3);
        for pixel in &self.pixels {
            rgb_data.push(pixel.r);
            rgb_data.push(pixel.g);
            rgb_data.push(pixel.b);
        }

        png_writer.write_image_data(&rgb_data)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ShaderUniforms {
    light_direction: Vec3,
    time: f32,
}

#[derive(Clone, Copy)]
struct FragmentInfo {
    world_pos: Vec3,
    normal: Vec3,
    view_dir: Vec3,
    light_intensity: f32,
    barycentric: [f32; 3],
    depth: f32,
}

#[derive(Clone, Copy)]
enum CelestialBody {
    Star,
    Rocky,
    GasGiant,
}

impl CelestialBody {
    fn label(&self) -> &'static str {
        match self {
            CelestialBody::Star => "star",
            CelestialBody::Rocky => "rocky",
            CelestialBody::GasGiant => "gas_giant",
        }
    }

    fn shade(&self, frag: &FragmentInfo, uniforms: &ShaderUniforms) -> Color {
        let rgb = match self {
            CelestialBody::Star => shade_star(frag, uniforms),
            CelestialBody::Rocky => shade_rocky(frag, uniforms),
            CelestialBody::GasGiant => shade_gas_giant(frag, uniforms),
        };
        vec3_to_color(rgb)
    }
}

fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a * (1.0 - t) + b * t
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn vec3_to_color(v: Vec3) -> Color {
    let clamped = v.clamp01();
    Color::RGB(
        (clamped.x * 255.0) as u8,
        (clamped.y * 255.0) as u8,
        (clamped.z * 255.0) as u8,
    )
}

fn fractal_noise(mut p: Vec3) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 0.5;
    let mut frequency = 2.0;

    for _ in 0..5 {
        let sample = ((p.x * frequency).sin()
            + (p.y * frequency * 1.3).sin()
            + (p.z * frequency * 0.7).sin())
            / 3.0;
        value += amplitude * (sample * 0.5 + 0.5);

        p = Vec3::new(
            p.y * 1.7 + 0.23,
            p.z * 1.9 - 0.41,
            p.x * 1.5 + 0.17,
        );
        amplitude *= 0.55;
        frequency *= 1.9;
    }

    value.clamp(0.0, 1.0)
}

fn shade_star(frag: &FragmentInfo, uniforms: &ShaderUniforms) -> Vec3 {
    let surface_normal = frag.normal;
    let longitude = surface_normal.z.atan2(surface_normal.x);
    let latitude = surface_normal.y;

    let swirl = (longitude * 5.0 + uniforms.time * 2.5).sin();
    let radial_waves = (latitude * 25.0 + uniforms.time * 4.0 + longitude * 3.0).sin();
    let convection =
        fractal_noise(surface_normal * 6.5 + Vec3::new(uniforms.time * 0.4, 0.0, 0.0));

    let dark = Vec3::new(0.85, 0.33, 0.08);
    let base = Vec3::new(1.0, 0.62, 0.2);
    let bright = Vec3::new(1.0, 0.9, 0.55);

    let dynamic = (swirl * 0.35 + radial_waves * 0.25).abs();
    let convection_mix = (convection * 0.7 + 0.2).clamp(0.0, 1.0);

    let mut color = lerp_vec3(dark, base, convection_mix);
    color = lerp_vec3(color, bright, dynamic);

    let glow = (frag.light_intensity * 0.6 + 0.4).powf(2.2);
    let rim = (1.0 - surface_normal.dot(frag.view_dir).max(0.0)).powf(3.0) * 0.35;

    color * glow + bright * rim
}

fn shade_rocky(frag: &FragmentInfo, uniforms: &ShaderUniforms) -> Vec3 {
    let animated_offset = Vec3::new(0.0, uniforms.time * 0.07, uniforms.time * 0.04);
    let continents = fractal_noise(frag.world_pos * 3.2 + animated_offset);
    let ridges = fractal_noise(frag.world_pos * 8.4 + Vec3::new(0.4, uniforms.time * 0.12, 0.2));
    let crater_noise =
        fractal_noise(frag.world_pos * 14.0 + Vec3::new(uniforms.time * 0.18, 0.6, 0.0));

    let lowlands = Vec3::new(0.32, 0.36, 0.28);
    let midlands = Vec3::new(0.48, 0.35, 0.26);
    let highlands = Vec3::new(0.68, 0.55, 0.43);
    let snow = Vec3::new(0.9, 0.94, 0.98);
    let crater_rim = Vec3::new(0.78, 0.66, 0.5);

    let land_mix = smoothstep(0.28, 0.72, continents);
    let mut color = lerp_vec3(lowlands, midlands, land_mix);

    let mountain_mask = smoothstep(0.55, 0.82, ridges);
    color = lerp_vec3(color, highlands, mountain_mask);

    let crater_mask = smoothstep(0.62, 0.84, crater_noise);
    color = lerp_vec3(color, crater_rim, crater_mask * 0.4);

    let polar_ice = smoothstep(0.55, 0.9, surface_falloff(frag.normal.y.abs()));
    color = lerp_vec3(color, snow, polar_ice * 0.65);

    let diffuse = 0.25 + 0.75 * frag.light_intensity;
    let rim = (1.0 - frag.normal.dot(frag.view_dir).max(0.0)).powf(4.0);

    color * diffuse + snow * rim * 0.18
}

fn shade_gas_giant(frag: &FragmentInfo, uniforms: &ShaderUniforms) -> Vec3 {
    let radius = frag.world_pos.length();

    if radius > 1.05 {
        let radial = (frag.world_pos.x * frag.world_pos.x + frag.world_pos.z * frag.world_pos.z)
            .sqrt()
            .max(1e-4);
        let ring_wave =
            ((radial - 1.15) * 40.0 + uniforms.time * 0.3).sin().abs().powf(1.2).clamp(0.0, 1.0);
        let subtle_noise = fractal_noise(Vec3::new(radial * 6.5, frag.world_pos.y * 12.0, 0.0));

        let base_ring = lerp_vec3(Vec3::new(0.85, 0.8, 0.72), Vec3::new(0.55, 0.52, 0.48), ring_wave);
        let ring_highlight = Vec3::new(0.95, 0.9, 0.82);
        let mut color = lerp_vec3(base_ring, ring_highlight, subtle_noise * 0.6);
        color *= 0.3 + 0.7 * frag.light_intensity;
        return color;
    }

    let lat = frag.normal.y;
    let band_pattern = (lat * 12.0 + uniforms.time * 0.45).sin();
    let turbulence = fractal_noise(
        frag.world_pos.mul_by_vec(Vec3::new(6.0, 10.0, 6.0))
            + Vec3::new(uniforms.time * 0.12, 0.0, uniforms.time * 0.08),
    );
    let swirl = (band_pattern * 0.6 + turbulence * 0.8 - 0.3).clamp(-1.0, 1.0);

    let band_mix = (swirl * 0.5 + 0.5).clamp(0.0, 1.0);
    let equator_emphasis = (1.0 - lat.abs()).powf(1.6).clamp(0.0, 1.0);

    let base_a = Vec3::new(0.93, 0.72, 0.54);
    let base_b = Vec3::new(0.68, 0.52, 0.74);
    let mut color = lerp_vec3(base_a, base_b, band_mix);
    color = lerp_vec3(color, Vec3::new(0.98, 0.82, 0.6), equator_emphasis * 0.4);

    let storms = fractal_noise(
        frag.world_pos.mul_by_vec(Vec3::new(9.5, 11.0, 9.5))
            + Vec3::new(uniforms.time * 0.18, uniforms.time * 0.3, 0.0),
    );
    let storm_mask = smoothstep(0.72, 0.93, storms) * 0.4;
    color = lerp_vec3(color, Vec3::new(0.95, 0.86, 0.75), storm_mask);

    let diffuse = 0.35 + 0.65 * frag.light_intensity;
    let rim = (1.0 - frag.normal.dot(frag.view_dir).max(0.0)).powf(3.0);

    color * diffuse + Vec3::new(1.0, 0.88, 0.73) * rim * 0.1
}

fn surface_falloff(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn rotate_y(v: Vec3, angle: f32) -> Vec3 {
    let cos = angle.cos();
    let sin = angle.sin();
    Vec3::new(v.x * cos + v.z * sin, v.y, -v.x * sin + v.z * cos)
}

fn rotate_x(v: Vec3, angle: f32) -> Vec3 {
    let cos = angle.cos();
    let sin = angle.sin();
    Vec3::new(v.x, v.y * cos - v.z * sin, v.y * sin + v.z * cos)
}

fn edge_function(a: &ProjectedVertex, b: &ProjectedVertex, px: f32, py: f32) -> f32 {
    (px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x)
}

fn draw_filled_triangle(
    buffer: &mut FrameBuffer,
    projected: &[ProjectedVertex; 3],
    world: &[Vec3; 3],
    normals: &[Vec3; 3],
    camera_space: &[Vec3; 3],
    body: CelestialBody,
    uniforms: &ShaderUniforms,
) {
    let min_x = projected
        .iter()
        .map(|v| v.x)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as i32;
    let max_x = projected
        .iter()
        .map(|v| v.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(buffer.width as f32 - 1.0) as i32;
    let min_y = projected
        .iter()
        .map(|v| v.y)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as i32;
    let max_y = projected
        .iter()
        .map(|v| v.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(buffer.height as f32 - 1.0) as i32;

    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge_function(&projected[0], &projected[1], projected[2].x, projected[2].y);
    if area.abs() < 1e-6 {
        return;
    }
    let inv_area = 1.0 / area;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            let w0 = edge_function(&projected[1], &projected[2], px, py);
            let w1 = edge_function(&projected[2], &projected[0], px, py);
            let w2 = edge_function(&projected[0], &projected[1], px, py);

            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0) {
                let bary0 = w0 * inv_area;
                let bary1 = w1 * inv_area;
                let bary2 = w2 * inv_area;

                let depth = bary0 * projected[0].z + bary1 * projected[1].z + bary2 * projected[2].z;
                let world_pos =
                    world[0] * bary0 + world[1] * bary1 + world[2] * bary2;
                let mut normal =
                    normals[0] * bary0 + normals[1] * bary1 + normals[2] * bary2;
                normal = normal.normalize();

                let camera_vec =
                    camera_space[0] * bary0 + camera_space[1] * bary1 + camera_space[2] * bary2;
                let view_dir = (-camera_vec).normalize();
                let light_intensity = normal.dot(uniforms.light_direction).max(0.0);

                let frag = FragmentInfo {
                    world_pos,
                    normal,
                    view_dir,
                    light_intensity,
                    barycentric: [bary0, bary1, bary2],
                    depth,
                };

                let color = body.shade(&frag, uniforms);
                buffer.try_set_pixel(x, y, color, depth);
            }
        }
    }
}

fn render_mesh(
    buffer: &mut FrameBuffer,
    mesh: &Mesh,
    body: CelestialBody,
    rotation: f32,
    uniforms: &ShaderUniforms,
) {
    let width = buffer.width as f32;
    let height = buffer.height as f32;
    let aspect = width / height;
    let fov = 60.0_f32.to_radians();
    let focal_length = 1.0 / (fov * 0.5).tan();
    let camera_offset = Vec3::new(0.0, 0.0, 3.0);

    for face in &mesh.faces {
        let mut projected = [ProjectedVertex::default(); 3];
        let mut world = [Vec3::default(); 3];
        let mut normals = [Vec3::default(); 3];
        let mut camera_space = [Vec3::default(); 3];
        let mut skip_face = false;

        for (i, &index) in face.iter().enumerate() {
            let vertex = mesh.vertices.get(index).copied().unwrap_or_default();
            let normal = mesh.vertex_normals.get(index).copied().unwrap_or_default();

            let rotated = rotate_y(rotate_x(vertex, BASE_TILT_X), rotation + BASE_ROTATION_Y);
            let rotated_normal = rotate_y(rotate_x(normal, BASE_TILT_X), rotation + BASE_ROTATION_Y);
            world[i] = rotated;
            normals[i] = rotated_normal.normalize();

            let cam_space = rotated + camera_offset;
            camera_space[i] = cam_space;

            if cam_space.z <= 0.1 {
                skip_face = true;
                break;
            }

            let ndc_x = (cam_space.x * focal_length) / (cam_space.z * aspect);
            let ndc_y = (cam_space.y * focal_length) / cam_space.z;

            let screen_x = ((ndc_x + 1.0) * 0.5) * (width - 1.0);
            let screen_y = ((1.0 - ndc_y) * 0.5) * (height - 1.0);

            projected[i] = ProjectedVertex {
                x: screen_x,
                y: screen_y,
                z: cam_space.z,
            };
        }

        if skip_face {
            continue;
        }

        let face_normal = (camera_space[1] - camera_space[0]).cross(camera_space[2] - camera_space[0]);
        if face_normal.z >= 0.0 {
            continue;
        }

        draw_filled_triangle(
            buffer,
            &projected,
            &world,
            &normals,
            &camera_space,
            body,
            uniforms,
        );
    }
}

fn mesh_for_body<'a>(body: CelestialBody, star: &'a Mesh, rocky: &'a Mesh, gas: &'a Mesh) -> &'a Mesh {
    match body {
        CelestialBody::Star => star,
        CelestialBody::Rocky => rocky,
        CelestialBody::GasGiant => gas,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let star_mesh = Mesh::load_from_obj(STAR_OBJ_PATH)?;
    let rocky_mesh = Mesh::load_from_obj(ROCKY_OBJ_PATH)?;
    let gas_mesh = Mesh::load_from_obj(GAS_GIANT_OBJ_PATH)?;

    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window("Lab 5 - Sistema Planetario", SCREEN_WIDTH as u32, SCREEN_HEIGHT as u32)
        .position_centered()
        .build()?;

    let mut canvas = window.into_canvas().accelerated().present_vsync().build()?;
    let mut event_pump = sdl_context.event_pump()?;
    let mut framebuffer = FrameBuffer::new(SCREEN_WIDTH, SCREEN_HEIGHT);

    let mut rotation = 0.0f32;
    let mut elapsed = 0.0f32;
    let mut previous_frame = Instant::now();
    let mut current_body = CelestialBody::Star;

    println!("Controles:");
    println!("  1 -> Estrella");
    println!("  2 -> Planeta rocoso");
    println!("  3 -> Gigante gaseoso (con anillos)");
    println!("  R -> Guardar render actual como PNG");
    println!("  ESC / Cerrar ventana -> Salir");

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::Num1),
                    ..
                }
                | Event::KeyDown {
                    keycode: Some(Keycode::Kp1),
                    ..
                } => current_body = CelestialBody::Star,
                Event::KeyDown {
                    keycode: Some(Keycode::Num2),
                    ..
                }
                | Event::KeyDown {
                    keycode: Some(Keycode::Kp2),
                    ..
                } => current_body = CelestialBody::Rocky,
                Event::KeyDown {
                    keycode: Some(Keycode::Num3),
                    ..
                }
                | Event::KeyDown {
                    keycode: Some(Keycode::Kp3),
                    ..
                } => current_body = CelestialBody::GasGiant,
                Event::KeyDown {
                    keycode: Some(Keycode::R),
                    ..
                } => {
                    let filename = format!("{}_frame.png", current_body.label());
                    framebuffer.save_png(&filename)?;
                    println!("Frame guardado como {}", filename);
                }
                _ => {}
            }
        }

        let now = Instant::now();
        let delta = now - previous_frame;
        previous_frame = now;
        let delta_secs = delta.as_secs_f32();

        rotation += delta_secs * 0.6;
        elapsed += delta_secs;

        framebuffer.clear(Color::RGB(8, 10, 22));

        let uniforms = ShaderUniforms {
            light_direction: Vec3::new(-0.4, 0.9, -1.0).normalize(),
            time: elapsed,
        };

        let mesh = mesh_for_body(current_body, &star_mesh, &rocky_mesh, &gas_mesh);
        render_mesh(&mut framebuffer, mesh, current_body, rotation, &uniforms);

        framebuffer.draw_to_canvas(&mut canvas)?;
        canvas.present();

        std::thread::sleep(Duration::from_millis(16));
    }

    Ok(())
}
