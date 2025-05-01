public class item {
    static class Item {
        private String description;
        private double price;

        public Item(String description, double price) {
            this.description = description;
            this.price = price;
        }

        public double getPrice() {
            return this.price;
        }

        public String getDescription() {
            return this.description;
        }
    }
}
