use bevy::asset::RenderAssetUsages;
use bevy::pbr::wireframe::{Wireframe, WireframePlugin};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use nalgebra::{Point3, Vector3};
use ply_rs::{parser, ply};
use poisson_reconstruction::marching_cubes::MeshBuffers;
use poisson_reconstruction::{PoissonReconstruction, Real};
use std::io::BufRead;
use std::path::Path;
use std::str::FromStr;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, PanOrbitCameraPlugin))
        .add_plugins(WireframePlugin)
        .add_systems(Startup, setup_camera_and_light)
        .add_systems(Startup, setup_scene)
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let point_cloud = parse_file("./assets/xiaojiejie2_pcd.ply", true);
    let surface = reconstruct_surface(&point_cloud);
    spawn_mesh(&mut commands, &mut meshes, &mut materials, surface);
    dbg!("Done");
}

fn setup_camera_and_light(mut commands: Commands) {
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 900000.0,
            range: 1000.,
            ..default()
        },
        transform: Transform::from_xyz(-100.0, 50.0, -100.0),
        ..default()
    });
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(-100.0, 25.0, -100.0)
                .looking_at(Vec3::new(0., 0., 0.), Vec3::Y),
            ..default()
        },
        PanOrbitCamera::default(),
    ));
}

fn spawn_mesh(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    points: MeshBuffers,
) {
    // Create the bevy mesh.
    let vertices: Vec<_> = points
        .vertices()
        .iter()
        .map(|pt| [pt.x as f32, pt.y as f32, pt.z as f32])
        .collect();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_indices(Indices::U32(points.indices().to_vec()));

    commands
        .spawn(PbrBundle {
            mesh: Mesh3d(meshes.add(mesh)),
            material: MeshMaterial3d(materials.add(Color::srgb(0.0, 1.0, 0.0))),
            transform: Transform::from_rotation(Quat::from_rotation_x(180.0f32.to_radians())),
            ..default()
        })
        .insert(Wireframe);
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
                "vertex" => {
                    vertex_list = vertex_parser
                        .read_payload_for_element(&mut f, &element, &header)
                        .unwrap();
                }
                _ => {}
            }
        }
        vertex_list
    } else {
        let mut result = vec![];
        for line in f.lines() {
            if let Ok(line) = line {
                let values: Vec<_> = line
                    .split_whitespace()
                    .map(|elt| f64::from_str(elt).unwrap())
                    .collect();
                result.push(VertexWithNormal {
                    pos: Point3::new(values[0], values[1], values[2]),
                    normal: Vector3::new(values[3], values[4], values[5]),
                });
            }
        }
        result
    }
}

fn reconstruct_surface(vertices: &[VertexWithNormal]) -> MeshBuffers {
    let points: Vec<_> = vertices.iter().map(|v| v.pos).collect();
    let normals: Vec<_> = vertices.iter().map(|v| v.normal).collect();

    dbg!("Running poisson.");
    let poisson = PoissonReconstruction::from_points_and_normals(&points, &normals, 0.0, 6, 6, 10);
    dbg!("Extracting vertices.");
    poisson.reconstruct_mesh_buffers()
}
