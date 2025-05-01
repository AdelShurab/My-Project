public class queue {
    static class Queue {
        private DLLOrder orders;

        public Queue() {
            this.orders = new DLLOrder();
        }

        public void addOrder(Order order) {
            orders.addOrder(order);
        }

        public Order removeOrder() {
            return orders.removeOrder();
        }

        public boolean isEmpty() {
            return orders.isEmpty();
        }
    }

}
