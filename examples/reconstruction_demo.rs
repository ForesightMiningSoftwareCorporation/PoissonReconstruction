use std::cell::RefCell;
use std::io::BufRead;
use std::path::Path;
use std::rc::Rc;
use kiss3d::window::Window;
use kiss3d::light::Light;
use poisson_reconstruction::{PoissonReconstruction, Real};
use nalgebra::{Point3, Vector3};
use ply_rs::{parser, ply};
use std::str::FromStr;
use kiss3d::camera::ArcBall;


pub fn main() {
    let mut window = Window::new("Kiss3d: cube");

    let point_cloud = parse_file("./assets/xiaojiejie2_pcd.ply", true);
    let surface = reconstruct_surface(&point_cloud);
    let mesh_points: Vec<_> = surface.iter().map(|pt| kiss3d::nalgebra::Point3::new(pt.x as f32, -pt.y as f32, pt.z as f32)).collect();

    // NOTE: kiss3d only support u16 index buffers, so we split into several meshes if
    // we have too many vertices.
    dbg!("Creating visuals");
    for mesh_part in mesh_points.chunks((u16::MAX as usize / 3) * 3) {
        let mesh_faces = (0..mesh_part.len() as u16 / 3).map(|i| kiss3d::nalgebra::Point3::new(i * 3, i * 3 + 1, i * 3 + 2))
            .collect();
        let mesh = kiss3d::resource::Mesh::new(mesh_part.to_vec(), mesh_faces, None, None, false);
        let mut c = window.add_mesh(Rc::new(RefCell::new(mesh)), kiss3d::nalgebra::Vector3::repeat(1.0));
        c.set_color(1.0, 1.0, 0.0);
        c.set_lines_width(2.0);
        c.set_lines_color(Some(kiss3d::nalgebra::Point3::new(0.0, 0.0, 0.0)));
    }
    dbg!("Done");

    let mut camera = ArcBall::new([-100.0, 25.0, -100.0].into(), [0.0, 0.0, 0.0].into());
    window.set_light(Light::StickToCamera);

    while window.render_with_camera(&mut camera) {
    }
}

#[derive(Default)]
struct VertexWithNormal {
    pos: Point3<Real>,
    normal: Vector3<Real>,
}

impl ply::PropertyAccess for VertexWithNormal {
    fn new() -> Self {
        Self::default()
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.pos.x = v as Real,
            ("y", ply::Property::Float(v)) => self.pos.y = v as Real,
            ("z", ply::Property::Float(v)) => self.pos.z = v as Real,
            ("nx", ply::Property::Float(v)) => self.normal.x = v as Real,
            ("ny", ply::Property::Float(v)) => self.normal.y = v as Real,
            ("nz", ply::Property::Float(v)) => self.normal.z = v as Real,
            _ => {}
        }
    }
}

fn parse_file(path: impl AsRef<Path>, ply: bool) -> Vec<VertexWithNormal> {
    let f = std::fs::File::open(path).unwrap();
    let mut f = std::io::BufReader::new(f);

    if ply {
        let vertex_parser = parser::Parser::<VertexWithNormal>::new();
        let header = vertex_parser.read_header(&mut f).unwrap();

        // Depending on the header, read the data into our structs..
        let mut vertex_list = Vec::new();
        for (_ignore_key, element) in &header.elements {
            // we could also just parse them in sequence, but the file format might change
            match element.name.as_ref() {
                "vertex" => { vertex_list = vertex_parser.read_payload_for_element(&mut f, &element, &header).unwrap(); },
                _ => {}
            }
        }
        vertex_list
    } else {
        let mut result = vec![];
        for line in f.lines() {
            if let Ok(line) = line {
                let values: Vec<_> = line.split_whitespace().map(|elt| f64::from_str(elt).unwrap()).collect();
                result.push(VertexWithNormal {
                    pos: Point3::new(values[0], values[1], values[2]),
                    normal: Vector3::new(values[3], values[4], values[5]),
                });
            }
        }
        result
    }
}

fn reconstruct_surface(vertices: &[VertexWithNormal]) -> Vec<Point3<Real>> {
    let points: Vec<_> = vertices.iter().map(|v| v.pos).collect();
    let normals: Vec<_> = vertices.iter().map(|v| v.normal).collect();

    dbg!("Running poisson.");
    let poisson = PoissonReconstruction::from_points_and_normals(
        &points, &normals, 0.0, 6, 6, 10
    );
    dbg!("Extracting vertices.");
    poisson.reconstruct_mesh()
}