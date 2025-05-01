public class DLLorder {
    class DLLOrder {
        NodeOrder head, tail;

        public void addOrder(Order order) {
            NodeOrder newOrder = new NodeOrder(order);
            if (head == null) {
                head = tail = newOrder;
            } else {
                tail.next = newOrder;
                newOrder.prev = tail;
                tail = newOrder;
            }
        }

    }
}