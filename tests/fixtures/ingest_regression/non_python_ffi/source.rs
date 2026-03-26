pub struct Integrator {
    position: f64,
    velocity: f64,
}

impl Integrator {
    pub fn step(&mut self, dt: f64) {
        self.position = self.position + self.velocity * dt;
    }

    pub fn get_position(&self) -> f64 {
        self.position
    }
}

