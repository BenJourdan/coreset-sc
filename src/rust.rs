// Some pure rust functions

#[allow(dead_code)]
pub fn rust_sum_as_string(a: usize, b: usize) -> String {
    (a + b).to_string()
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_sum_as_string() {
        assert_eq!(rust_sum_as_string(1, 2), "3");
    }
}
